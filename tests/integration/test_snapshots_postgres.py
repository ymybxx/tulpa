from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.db.facts import FactStore
from fish_memory.db.fishes import FishStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.sessions import SessionStore
from fish_memory.db.snapshots import SnapshotStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.facts.models import ExtractedFact
from fish_memory.models import MessageStatus
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    SnapshotItemMutation,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
EVALUATION_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
EVALUATION_OWNER = UUID("00000000-0000-0000-0000-000000000003")
NOW = datetime(2026, 7, 24, tzinfo=UTC)
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)


@pytest.mark.asyncio
async def test_normalized_reads_keep_all_items_in_support_time_and_id_order(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches database age filtering or an unstable item order in either read API."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized-read-order")
    source = await MessagesStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).insert_user_message(fish_id, "create session")
    snapshot_id = await postgres_pool.fetchval(
        """
        INSERT INTO memory_snapshots (
            fish_id, session_id, snapshot_type, content, through_fact_sequence
        )
        VALUES ($1, $2, 'dynamic', NULL, 3)
        RETURNING id
        """,
        fish_id,
        source.session_id,
    )
    later_fact = uuid4()
    old_first_fact = uuid4()
    old_second_fact = uuid4()
    old_time = NOW - timedelta(days=365)
    for summary, fact_id, supported_at in (
        ("later", later_fact, NOW),
        ("old-first", old_first_fact, old_time),
        ("old-second", old_second_fact, old_time),
    ):
        await postgres_pool.execute(
            """
            INSERT INTO memory_snapshot_items (
                snapshot_id,
                summary,
                support_fact_ids,
                lineage_fact_ids,
                last_supported_at
            )
            VALUES ($1, $2, $3::uuid[], $3::uuid[], $4)
            """,
            snapshot_id,
            summary,
            [fact_id],
            supported_at,
        )

    store = SnapshotStore(postgres_pool)
    snapshot = await store.load_normalized_dynamic_snapshot(
        fish_id,
        source.session_id,
    )
    overviews = await store.list_normalized_current_overviews(fish_id)

    assert snapshot is not None
    assert [item.summary for item in snapshot.items] == [
        "old-first",
        "old-second",
        "later",
    ]
    assert [item.summary for item in overviews[0].items] == [
        "old-first",
        "old-second",
        "later",
    ]


@pytest.mark.asyncio
async def test_normalized_reads_never_fallback_to_v1_json_content(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches a compatibility fallback that would make legacy Fish conversational."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="legacy-read-boundary")
    source = await MessagesStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).insert_user_message(fish_id, "create legacy session")
    await postgres_pool.execute(
        "UPDATE fishes SET snapshot_storage_version = 1 WHERE id = $1",
        fish_id,
    )
    await postgres_pool.execute(
        """
        INSERT INTO memory_snapshots (
            fish_id, session_id, snapshot_type, content, through_fact_sequence
        )
        VALUES (
            $1,
            $2,
            'dynamic',
            '{"items":[{"category":"state","status":"active",'
            '"text":"legacy content","sourceFactIds":[],"lastSupportedAt":'
            '"2026-07-24T00:00:00Z","deprecatedReason":null}]}'::jsonb,
            3
        )
        """,
        fish_id,
        source.session_id,
    )
    store = SnapshotStore(postgres_pool)

    dynamic = await store.load_normalized_dynamic_snapshot(
        fish_id,
        source.session_id,
    )
    static = await store.load_normalized_static_profile(fish_id)
    overviews = await store.list_normalized_current_overviews(fish_id)

    assert dynamic is None
    assert static is None
    assert overviews == []
    assert (
        await postgres_pool.fetchval(
            """
            SELECT content->'items'->0->>'text'
            FROM memory_snapshots
            WHERE fish_id = $1 AND session_id = $2
            """,
            fish_id,
            source.session_id,
        )
        == "legacy content"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_count", "first_turn"),
    [(5, 1), (6, 2)],
)
async def test_load_recent_turns_uses_a_real_five_turn_window(
    postgres_pool: asyncpg.Pool,
    turn_count: int,
    first_turn: int,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    snapshots = SnapshotStore(postgres_pool)
    session_id: UUID | None = None
    anchor_id: UUID | None = None
    for number in range(1, turn_count + 1):
        user = await messages.insert_user_message(
            FISH_ID,
            f"user-{number}",
            session_id=session_id,
        )
        session_id = user.session_id
        assistant = await messages.insert_generating_answer(
            FISH_ID,
            session_id,
            user.sequence,
        )
        if number == turn_count:
            await messages.update_consumed_offset(assistant.id, FISH_ID, 4)
            content = "keep hidden"
        else:
            content = f"assistant-{number}"
        await messages.finish_answer(
            assistant.id,
            FISH_ID,
            content=content,
            requested_status=MessageStatus.COMPLETED,
        )
        anchor_id = assistant.id

    await messages.insert_user_message(FISH_ID, "future", session_id=session_id)
    assert session_id is not None
    assert anchor_id is not None

    turns = await snapshots.load_recent_turns_for_fish(FISH_ID, anchor_id)

    expected = tuple(
        message
        for number in range(first_turn, turn_count + 1)
        for message in (
            {"role": "user", "content": f"user-{number}"},
            {
                "role": "assistant",
                "content": "keep" if number == turn_count else f"assistant-{number}",
            },
        )
    )
    assert turns == expected
    assert "future" not in [turn["content"] for turn in turns]


@pytest.mark.asyncio
async def test_load_recent_turns_rejects_cross_session_and_missing_anchors(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    snapshots = SnapshotStore(postgres_pool)
    await messages.insert_user_message(FISH_ID, "primary")
    foreign = await messages.insert_user_message(EVALUATION_FISH_ID, "foreign")

    with pytest.raises(ValueError, match="does not belong"):
        await snapshots.load_recent_turns_for_fish(FISH_ID, foreign.id)
    with pytest.raises(ValueError, match="does not belong"):
        await snapshots.load_recent_turns_for_fish(FISH_ID, uuid4())


@pytest.mark.asyncio
async def test_load_support_facts_preserves_order_and_includes_invalidated_rows(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches confirmation evidence being filtered or reordered in real SQL."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="support-fact-loader")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    source = await messages.insert_user_message(fish_id, "两条支持事实")
    await messages.insert_user_message(
        fish_id,
        "保持前一条消息稳定",
        session_id=source.session_id,
    )
    pending = (await facts.list_pending_messages(limit=10))[0]
    assert pending.message.id == source.id
    await facts.persist_extraction(
        pending,
        [
            ExtractedFact(
                text="第一条旧支持",
                occurred_start=None,
                occurred_end=None,
                entities=("第一条",),
            ),
            ExtractedFact(
                text="第二条当前支持",
                occurred_start=None,
                occurred_end=None,
                entities=("第二条",),
            ),
        ],
        [(0.25,) * 1024, (0.5,) * 1024],
        EMBEDDING_SPACE,
    )
    rows = await postgres_pool.fetch(
        "SELECT id FROM facts WHERE source_message_id = $1 ORDER BY fact_index",
        source.id,
    )
    first_id, second_id = (row["id"] for row in rows)
    await postgres_pool.execute(
        "UPDATE facts SET invalidated_at = now() WHERE id = $1",
        first_id,
    )

    loaded = await SnapshotStore(postgres_pool).load_support_facts(
        fish_id,
        (second_id, first_id),
    )

    assert [fact.id for fact in loaded] == [second_id, first_id]
    assert [fact.text for fact in loaded] == [
        "第二条当前支持",
        "第一条旧支持",
    ]
    with pytest.raises(ValueError, match="do not belong"):
        await SnapshotStore(postgres_pool).load_support_facts(
            EVALUATION_FISH_ID,
            (first_id,),
        )


@pytest.mark.asyncio
async def test_snapshots_wait_until_fact_extraction_reaches_stable_messages(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized-stable-window")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    await messages.insert_user_message(fish_id, "第一条稳定消息")
    await messages.insert_user_message(fish_id, "第二条稳定消息")
    await messages.insert_user_message(fish_id, "最新消息暂不处理")

    first_pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        first_pending,
        [
            ExtractedFact(
                text="该用户提到第一条稳定消息",
                occurred_start=None,
                occurred_end=None,
                entities=("稳定消息",),
            )
        ],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )

    assert (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20) == []
    )
    assert (
        await snapshots.list_pending_normalized_static_batches(
            limit=10,
            batch_size=20,
        )
        == []
    )

    second_pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        second_pending,
        [
            ExtractedFact(
                text="该用户提到第二条稳定消息",
                occurred_start=None,
                occurred_end=None,
                entities=("稳定消息",),
            )
        ],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )

    dynamic_batches = await snapshots.list_pending_normalized_batches(
        limit=10,
        batch_size=20,
    )
    static_batches = await snapshots.list_pending_normalized_static_batches(
        limit=10,
        batch_size=20,
    )

    assert len(dynamic_batches) == 1
    assert len(static_batches) == 1
    assert [fact.text for fact in dynamic_batches[0].facts] == [
        "该用户提到第一条稳定消息",
        "该用户提到第二条稳定消息",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_is_first", [False, True])
async def test_unplayed_background_message_does_not_block_ready_snapshots(
    postgres_pool: asyncpg.Pool,
    waiting_is_first: bool,
) -> None:
    """Unheard output must neither enter memory nor hold another session hostage."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE).create(
        name="snapshot-consumption-frontier"
    )
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    seed = await messages.insert_user_message(fish_id, "已确认的旧会话内容")
    async with postgres_pool.acquire() as connection:
        waiting = await messages.insert_background_message_on_connection(
            connection,
            fish_id,
            seed.session_id,
            seed.sequence,
            "已播部分未播部分",
        )
    if waiting_is_first:
        # Cancellation can leave a session whose first remaining row is unplayed.
        await postgres_pool.execute("DELETE FROM messages WHERE id = $1", seed.id)
    else:
        pending = (await facts.list_pending_messages(10))[0]
        await facts.persist_extraction(
            pending,
            [ExtractedFact("已确认的旧会话内容", None, None, ())],
            [(0.25,) * 1024],
            EMBEDDING_SPACE,
        )
    await messages.insert_user_message(
        fish_id, "等待消息之后的内容", session_id=seed.session_id
    )

    current_session = await SessionStore(postgres_pool).create(fish_id)
    current = await messages.insert_user_message(
        fish_id, "新会话已确认内容", session_id=current_session.id
    )
    await messages.insert_user_message(
        fish_id, "最新消息", session_id=current.session_id
    )
    pending_messages = await facts.list_pending_messages(10)
    assert [p.message.id for p in pending_messages] == [current.id]
    await facts.persist_extraction(
        pending_messages[0],
        [ExtractedFact("新会话已确认内容", None, None, ())],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )

    dynamic = await snapshots.list_pending_normalized_batches(10, 100)
    static = await snapshots.list_pending_normalized_static_batches(10, 100)
    assert {batch.session_id for batch in dynamic} == (
        {current.session_id}
        if waiting_is_first
        else {seed.session_id, current.session_id}
    )
    assert len(static) == 1
    assert {fact.text for fact in static[0].facts} == (
        {"新会话已确认内容"}
        if waiting_is_first
        else {"已确认的旧会话内容", "新会话已确认内容"}
    )
    # Advance real snapshot cursors, then verify a late acknowledgement is not lost.
    for batch in dynamic:
        assert await snapshots.persist_mutations(batch, ())
    assert await snapshots.persist_mutations(static[0], ())
    through_before = static[0].target_fact_sequence
    assert await snapshots.list_pending_normalized_static_batches(10, 100) == []
    row = await postgres_pool.fetchrow(
        "SELECT consumed_offset, fact_extraction_status FROM messages WHERE id = $1",
        waiting.id,
    )
    assert dict(row) == {
        "consumed_offset": 0,
        "fact_extraction_status": "waiting_consumption",
    }

    await messages.finalize_background_message_consumption(
        waiting.id, fish_id, consumed_offset=4, interrupted=True
    )
    late = (await facts.list_pending_messages(10))[0]
    assert late.message.id == waiting.id
    assert late.message.content == "已播部分"
    await facts.persist_extraction(
        late,
        [ExtractedFact("仅已播部分的事实", None, None, ())],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )
    late_static = await snapshots.list_pending_normalized_static_batches(10, 100)
    assert len(late_static) == 1
    assert [fact.text for fact in late_static[0].facts] == ["仅已播部分的事实"]
    assert late_static[0].target_fact_sequence > through_before
    late_dynamic = await snapshots.list_pending_normalized_batches(10, 100)
    assert [batch.session_id for batch in late_dynamic] == [seed.session_id]
    assert [fact.text for fact in late_dynamic[0].facts] == ["仅已播部分的事实"]


@pytest.mark.asyncio
async def test_normalized_mutations_preserve_identity_and_isolate_scopes(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches normalized writes that replace IDs, lose lineage, or mix scopes."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    first_source = await messages.insert_user_message(
        fish_id,
        "两个初始主题",
        created_at=NOW,
    )
    second_source = await messages.insert_user_message(
        fish_id,
        "三个后续主题",
        session_id=first_source.session_id,
        created_at=NOW.replace(minute=1),
    )
    first_pending = (await facts.list_pending_messages(limit=10))[0]
    assert first_pending.message.id == first_source.id
    await facts.persist_extraction(
        first_pending,
        [
            ExtractedFact(
                text="用户保留第一主题",
                occurred_start=None,
                occurred_end=None,
                entities=("第一主题",),
            ),
            ExtractedFact(
                text="用户保留第二主题",
                occurred_start=None,
                occurred_end=None,
                entities=("第二主题",),
            ),
        ],
        [(0.25,) * 1024, (0.5,) * 1024],
        EMBEDDING_SPACE,
    )

    initial_dynamic = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    initial_static = (
        await snapshots.list_pending_normalized_static_batches(
            limit=10,
            batch_size=20,
        )
    )[0]
    first_fact, second_fact = initial_dynamic.facts
    dynamic_creates = tuple(
        SnapshotItemMutation(
            action="create",
            item_id=None,
            basis=None,
            summary=summary,
            support_fact_ids=(fact.id,),
            lineage_fact_ids=(fact.id,),
            new_support_fact_ids=(fact.id,),
            removed_support_fact_ids=(),
            last_supported_at=fact.mentioned_at,
        )
        for summary, fact in (
            ("第一主题初始状态", first_fact),
            ("第二主题初始状态", second_fact),
        )
    )
    static_create = SnapshotItemMutation(
        action="create",
        item_id=None,
        basis=None,
        summary="用户长期保留第一主题",
        support_fact_ids=(first_fact.id,),
        lineage_fact_ids=(first_fact.id,),
        new_support_fact_ids=(first_fact.id,),
        removed_support_fact_ids=(),
        last_supported_at=first_fact.mentioned_at,
    )

    assert await snapshots.persist_mutations(initial_dynamic, dynamic_creates)
    assert await snapshots.persist_mutations(initial_static, (static_create,))
    dynamic_before = await snapshots.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    static_profile = await snapshots.load_normalized_static_profile(fish_id)
    assert dynamic_before is not None
    assert static_profile is not None
    assert len(dynamic_before.items) == 2
    assert len(static_profile.items) == 1
    assert {item.id for item in dynamic_before.items}.isdisjoint(
        item.id for item in static_profile.items
    )

    await messages.insert_user_message(
        fish_id,
        "未来消息",
        session_id=first_source.session_id,
        created_at=NOW.replace(minute=2),
    )
    second_pending = (await facts.list_pending_messages(limit=10))[0]
    assert second_pending.message.id == second_source.id
    await facts.persist_extraction(
        second_pending,
        [
            ExtractedFact(
                text="第一主题已更新",
                occurred_start=None,
                occurred_end=None,
                entities=("第一主题",),
            ),
            ExtractedFact(
                text="第二主题增加支持",
                occurred_start=None,
                occurred_end=None,
                entities=("第二主题",),
            ),
            ExtractedFact(
                text="出现第三主题",
                occurred_start=None,
                occurred_end=None,
                entities=("第三主题",),
            ),
        ],
        [(0.25,) * 1024, (0.5,) * 1024, (0.75,) * 1024],
        EMBEDDING_SPACE,
    )
    update_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    update_by_summary = {item.summary: item for item in update_batch.snapshot.items}
    first_item = update_by_summary["第一主题初始状态"]
    second_item = update_by_summary["第二主题初始状态"]
    first_new, second_new, third_new = update_batch.facts
    first_update = SnapshotItemMutation(
        action="update",
        item_id=first_item.id,
        basis="第一主题的新事实替换旧支持",
        summary="第一主题当前状态",
        support_fact_ids=(first_new.id,),
        lineage_fact_ids=(*first_item.lineage_fact_ids, first_new.id),
        new_support_fact_ids=(first_new.id,),
        removed_support_fact_ids=first_item.support_fact_ids,
        last_supported_at=first_new.mentioned_at,
    )
    second_update = SnapshotItemMutation(
        action="update",
        item_id=second_item.id,
        basis="第二主题获得额外支持",
        summary="第二主题当前状态",
        support_fact_ids=(*second_item.support_fact_ids, second_new.id),
        lineage_fact_ids=(*second_item.lineage_fact_ids, second_new.id),
        new_support_fact_ids=(second_new.id,),
        removed_support_fact_ids=(),
        last_supported_at=second_new.mentioned_at,
    )
    third_create = SnapshotItemMutation(
        action="create",
        item_id=None,
        basis="第三主题独立存在",
        summary="第三主题当前状态",
        support_fact_ids=(third_new.id,),
        lineage_fact_ids=(third_new.id,),
        new_support_fact_ids=(third_new.id,),
        removed_support_fact_ids=(),
        last_supported_at=third_new.mentioned_at,
    )

    assert await snapshots.persist_mutations(
        update_batch,
        (third_create, second_update, first_update),
    )

    dynamic_after = await snapshots.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    assert dynamic_after is not None
    after_by_id = {item.id: item for item in dynamic_after.items}
    assert after_by_id[first_item.id].summary == "第一主题当前状态"
    assert after_by_id[first_item.id].support_fact_ids == (first_new.id,)
    assert after_by_id[first_item.id].lineage_fact_ids == (
        first_fact.id,
        first_new.id,
    )
    assert after_by_id[first_item.id].item_version == 2
    assert after_by_id[second_item.id].item_version == 2
    assert len(dynamic_after.items) == 3
    headers = await postgres_pool.fetch(
        """
        SELECT id, snapshot_type, content, version
        FROM memory_snapshots
        WHERE fish_id = $1
        ORDER BY snapshot_type
        """,
        fish_id,
    )
    assert [(row["snapshot_type"], row["version"]) for row in headers] == [
        ("dynamic", 2),
        ("static", 1),
    ]
    assert all(row["content"] is None for row in headers)
    dynamic_header_id = next(
        row["id"] for row in headers if row["snapshot_type"] == "dynamic"
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_revisions WHERE snapshot_id = $1",
            dynamic_header_id,
        )
        == 2
    )
    assert (
        await postgres_pool.fetchval(
            """
        SELECT count(*)
        FROM memory_snapshot_item_versions AS item_version
        JOIN memory_snapshot_items AS item
          ON item.id = item_version.snapshot_item_id
        WHERE item.snapshot_id = $1
        """,
            dynamic_header_id,
        )
        == 5
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*)
            FROM facts
            WHERE fish_id = $1 AND invalidated_at IS NOT NULL
            """,
            fish_id,
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_versions WHERE fish_id = $1",
            fish_id,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_normalized_no_mutation_and_header_conflict_are_noop_safe(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches no-op batches that add revisions or stale batches that advance state."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized-noop")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    first = await messages.insert_user_message(fish_id, "first", created_at=NOW)
    second = await messages.insert_user_message(
        fish_id,
        "second",
        session_id=first.session_id,
        created_at=NOW.replace(minute=1),
    )
    pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        pending,
        [
            ExtractedFact(
                text="first fact",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            )
        ],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )
    initial_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]

    assert await snapshots.persist_mutations(initial_batch, ())
    assert not await snapshots.persist_mutations(
        initial_batch,
        (
            SnapshotItemMutation(
                action="create",
                item_id=None,
                basis=None,
                summary="stale create",
                support_fact_ids=(initial_batch.facts[0].id,),
                lineage_fact_ids=(initial_batch.facts[0].id,),
                new_support_fact_ids=(initial_batch.facts[0].id,),
                removed_support_fact_ids=(),
                last_supported_at=initial_batch.facts[0].mentioned_at,
            ),
        ),
    )

    await messages.insert_user_message(
        fish_id,
        "future",
        session_id=first.session_id,
        created_at=NOW.replace(minute=2),
    )
    second_pending = (await facts.list_pending_messages(limit=10))[0]
    assert second_pending.message.id == second.id
    await facts.persist_extraction(
        second_pending,
        [
            ExtractedFact(
                text="second fact",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            )
        ],
        [(0.5,) * 1024],
        EMBEDDING_SPACE,
    )
    existing_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    assert existing_batch.snapshot.version == 1
    assert await snapshots.persist_mutations(existing_batch, ())

    header = await postgres_pool.fetchrow(
        """
        SELECT id, version, through_fact_sequence
        FROM memory_snapshots
        WHERE fish_id = $1 AND session_id = $2 AND snapshot_type = 'dynamic'
        """,
        fish_id,
        first.session_id,
    )
    assert header is not None
    assert header["version"] == 1
    assert header["through_fact_sequence"] == existing_batch.target_fact_sequence
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_revisions WHERE snapshot_id = $1",
            header["id"],
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_items WHERE snapshot_id = $1",
            header["id"],
        )
        == 0
    )
    assert await snapshots.list_pending_normalized_batches(10, 20) == []


@pytest.mark.asyncio
async def test_normalized_write_failure_rolls_back_header_items_revision_and_progress(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches a transaction that commits partial normalized state on audit failure."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized-rollback")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    first = await messages.insert_user_message(fish_id, "first", created_at=NOW)
    second = await messages.insert_user_message(
        fish_id,
        "second",
        session_id=first.session_id,
        created_at=NOW.replace(minute=1),
    )
    pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        pending,
        [
            ExtractedFact(
                text="initial fact",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            )
        ],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )
    initial_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    initial_fact = initial_batch.facts[0]
    assert await snapshots.persist_mutations(
        initial_batch,
        (
            SnapshotItemMutation(
                action="create",
                item_id=None,
                basis=None,
                summary="initial summary",
                support_fact_ids=(initial_fact.id,),
                lineage_fact_ids=(initial_fact.id,),
                new_support_fact_ids=(initial_fact.id,),
                removed_support_fact_ids=(),
                last_supported_at=initial_fact.mentioned_at,
            ),
        ),
    )

    await messages.insert_user_message(
        fish_id,
        "future",
        session_id=first.session_id,
        created_at=NOW.replace(minute=2),
    )
    second_pending = (await facts.list_pending_messages(limit=10))[0]
    assert second_pending.message.id == second.id
    await facts.persist_extraction(
        second_pending,
        [
            ExtractedFact(
                text="replacement fact",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            )
        ],
        [(0.5,) * 1024],
        EMBEDDING_SPACE,
    )
    update_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    old_item = update_batch.snapshot.items[0]
    new_fact = update_batch.facts[0]
    before = await postgres_pool.fetchrow(
        """
        SELECT
            snapshot.id,
            snapshot.version,
            snapshot.through_fact_sequence,
            item.summary,
            item.item_version,
            progress.through_sequence,
            (SELECT count(*) FROM memory_snapshot_revisions
             WHERE snapshot_id = snapshot.id) AS revision_count,
            (SELECT count(*) FROM memory_snapshot_item_versions
             WHERE snapshot_item_id = item.id) AS item_version_count
        FROM memory_snapshots AS snapshot
        JOIN memory_snapshot_items AS item ON item.snapshot_id = snapshot.id
        JOIN session_background_task_progress AS progress
          ON progress.session_id = snapshot.session_id
         AND progress.task_name = 'dynamic_snapshot'
        WHERE snapshot.fish_id = $1
          AND snapshot.session_id = $2
          AND snapshot.snapshot_type = 'dynamic'
        """,
        fish_id,
        first.session_id,
    )
    assert before is not None
    await postgres_pool.execute(
        """
        CREATE FUNCTION fail_normalized_item_version() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'injected normalized item version failure';
        END;
        $$
        """
    )
    await postgres_pool.execute(
        """
        CREATE TRIGGER fail_normalized_item_version_trigger
        BEFORE INSERT ON memory_snapshot_item_versions
        FOR EACH ROW EXECUTE FUNCTION fail_normalized_item_version()
        """
    )

    with pytest.raises(
        asyncpg.PostgresError,
        match="injected normalized item version failure",
    ):
        await snapshots.persist_mutations(
            update_batch,
            (
                SnapshotItemMutation(
                    action="update",
                    item_id=old_item.id,
                    basis="replace old support",
                    summary="replacement summary",
                    support_fact_ids=(new_fact.id,),
                    lineage_fact_ids=(*old_item.lineage_fact_ids, new_fact.id),
                    new_support_fact_ids=(new_fact.id,),
                    removed_support_fact_ids=old_item.support_fact_ids,
                    last_supported_at=new_fact.mentioned_at,
                ),
            ),
        )

    after = await postgres_pool.fetchrow(
        """
        SELECT
            snapshot.id,
            snapshot.version,
            snapshot.through_fact_sequence,
            item.summary,
            item.item_version,
            progress.through_sequence,
            (SELECT count(*) FROM memory_snapshot_revisions
             WHERE snapshot_id = snapshot.id) AS revision_count,
            (SELECT count(*) FROM memory_snapshot_item_versions
             WHERE snapshot_item_id = item.id) AS item_version_count
        FROM memory_snapshots AS snapshot
        JOIN memory_snapshot_items AS item ON item.snapshot_id = snapshot.id
        JOIN session_background_task_progress AS progress
          ON progress.session_id = snapshot.session_id
         AND progress.task_name = 'dynamic_snapshot'
        WHERE snapshot.fish_id = $1
          AND snapshot.session_id = $2
          AND snapshot.snapshot_type = 'dynamic'
        """,
        fish_id,
        first.session_id,
    )
    assert dict(after or {}) == dict(before)


@pytest.mark.asyncio
async def test_normalized_candidates_are_v2_and_background_owner_isolated(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches candidate queries that leak legacy or another owner's work."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_store = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    normal_id = await fish_store.create(name="normal-v2")
    evaluation_id = await fish_store.create(name="evaluation-v2")
    await postgres_pool.execute(
        "UPDATE fishes SET background_owner = $2 WHERE id = $1",
        evaluation_id,
        EVALUATION_OWNER,
    )
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    sources = []
    for fish_id in (normal_id, evaluation_id):
        source = await messages.insert_user_message(fish_id, "stable")
        await messages.insert_user_message(
            fish_id,
            "latest",
            session_id=source.session_id,
        )
        sources.append(source)
        await postgres_pool.execute(
            """
            INSERT INTO facts (
                id, fish_id, session_id, source_message_id,
                fact_index, text, mentioned_at, entities
            )
            VALUES ($1, $2, $3, $4, 0, 'fact', now(), '[]'::jsonb)
            """,
            uuid4(),
            fish_id,
            source.session_id,
            source.id,
        )
        await postgres_pool.execute(
            """
            INSERT INTO session_background_task_progress (
                session_id, task_name, through_sequence
            )
            VALUES ($1, 'fact_extraction', $2)
            """,
            source.session_id,
            source.sequence,
        )

    normal = SnapshotStore(postgres_pool)
    evaluation = SnapshotStore(postgres_pool, background_owner=EVALUATION_OWNER)
    normal_batches = await normal.list_pending_normalized_batches(10, 10)
    evaluation_batches = await evaluation.list_pending_normalized_static_batches(
        10,
        10,
    )

    assert [batch.fish_id for batch in normal_batches] == [normal_id]
    assert [batch.fish_id for batch in evaluation_batches] == [evaluation_id]
    assert all(
        batch.fish_id not in (FISH_ID, EVALUATION_FISH_ID) for batch in normal_batches
    )


@pytest.mark.asyncio
async def test_same_version_batches_cas_through_sequence_before_noop_or_mutation(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches old same-version batches committing after a newer no-op batch."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=EMBEDDING_SPACE,
    ).create(name="normalized-watermark-cas")
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    snapshots = SnapshotStore(postgres_pool)
    source = await messages.insert_user_message(fish_id, "stable", created_at=NOW)
    await messages.insert_user_message(
        fish_id,
        "latest",
        session_id=source.session_id,
        created_at=NOW.replace(minute=1),
    )
    pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        pending,
        [
            ExtractedFact(
                text="initial fact",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            )
        ],
        [(0.25,) * 1024],
        EMBEDDING_SPACE,
    )
    initial_batch = (
        await snapshots.list_pending_normalized_batches(limit=10, batch_size=20)
    )[0]
    initial_fact = initial_batch.facts[0]
    assert await snapshots.persist_mutations(
        initial_batch,
        (
            SnapshotItemMutation(
                action="create",
                item_id=None,
                basis=None,
                summary="initial summary",
                support_fact_ids=(initial_fact.id,),
                lineage_fact_ids=(initial_fact.id,),
                new_support_fact_ids=(initial_fact.id,),
                removed_support_fact_ids=(),
                last_supported_at=initial_fact.mentioned_at,
            ),
        ),
    )
    aggregate = await snapshots.load_normalized_dynamic_snapshot(
        fish_id,
        source.session_id,
    )
    assert aggregate is not None
    older = replace(
        initial_batch,
        snapshot=aggregate,
        target_fact_sequence=aggregate.through_fact_sequence + 1,
    )
    newer = replace(
        initial_batch,
        snapshot=aggregate,
        target_fact_sequence=aggregate.through_fact_sequence + 2,
    )

    assert await snapshots.persist_mutations(newer, ())
    assert not await snapshots.persist_mutations(older, ())
    assert not await snapshots.persist_mutations(
        older,
        (
            SnapshotItemMutation(
                action="create",
                item_id=None,
                basis=None,
                summary="stale mutation",
                support_fact_ids=(initial_fact.id,),
                lineage_fact_ids=(initial_fact.id,),
                new_support_fact_ids=(initial_fact.id,),
                removed_support_fact_ids=(),
                last_supported_at=initial_fact.mentioned_at,
            ),
        ),
    )

    header = await postgres_pool.fetchrow(
        """
        SELECT id, version, through_fact_sequence
        FROM memory_snapshots
        WHERE fish_id = $1 AND session_id = $2 AND snapshot_type = 'dynamic'
        """,
        fish_id,
        source.session_id,
    )
    assert header is not None
    assert header["version"] == aggregate.version
    assert header["through_fact_sequence"] == newer.target_fact_sequence
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_revisions WHERE snapshot_id = $1",
            header["id"],
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_items WHERE snapshot_id = $1",
            header["id"],
        )
        == 1
    )


@pytest.mark.asyncio
async def test_dynamic_persistence_rejects_session_owned_by_another_fish_and_owner(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches Dynamic headers or progress attached to another Fish's Session."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_store = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    normal_id = await fish_store.create(name="normal-owner")
    evaluation_id = await fish_store.create(name="evaluation-owner")
    await postgres_pool.execute(
        "UPDATE fishes SET background_owner = $2 WHERE id = $1",
        evaluation_id,
        EVALUATION_OWNER,
    )
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    source = await messages.insert_user_message(evaluation_id, "stable")
    await messages.insert_user_message(
        evaluation_id,
        "latest",
        session_id=source.session_id,
    )
    fact_id = uuid4()
    await postgres_pool.execute(
        """
        INSERT INTO facts (
            id, fish_id, session_id, source_message_id,
            fact_index, text, mentioned_at, entities
        )
        VALUES ($1, $2, $3, $4, 0, 'fact', now(), '[]'::jsonb)
        """,
        fact_id,
        evaluation_id,
        source.session_id,
        source.id,
    )
    await postgres_pool.execute(
        """
        INSERT INTO session_background_task_progress (
            session_id, task_name, through_sequence
        )
        VALUES ($1, 'fact_extraction', $2)
        """,
        source.session_id,
        source.sequence,
    )
    evaluation = SnapshotStore(postgres_pool, background_owner=EVALUATION_OWNER)
    evaluation_batch = (await evaluation.list_pending_normalized_batches(10, 10))[0]
    cross_fish_batch = replace(evaluation_batch, fish_id=normal_id)
    normal = SnapshotStore(postgres_pool)

    assert not await normal.persist_mutations(cross_fish_batch, ())
    assert (
        await postgres_pool.fetchval(
            """
        SELECT count(*)
        FROM memory_snapshots
        WHERE fish_id = $1 AND session_id = $2
        """,
            normal_id,
            source.session_id,
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            """
        SELECT count(*)
        FROM session_background_task_progress
        WHERE session_id = $1 AND task_name = 'dynamic_snapshot'
        """,
            source.session_id,
        )
        == 0
    )

    header = await postgres_pool.fetchrow(
        """
        INSERT INTO memory_snapshots (
            fish_id, session_id, snapshot_type, content, through_fact_sequence
        )
        VALUES ($1, $2, 'dynamic', NULL, 0)
        RETURNING id, version
        """,
        normal_id,
        source.session_id,
    )
    assert header is not None
    existing_cross_fish_batch = replace(
        cross_fish_batch,
        snapshot=NormalizedDynamicSnapshot(
            items=(),
            through_fact_sequence=0,
            version=header["version"],
        ),
    )
    mutation = SnapshotItemMutation(
        action="create",
        item_id=None,
        basis=None,
        summary="cross-fish mutation",
        support_fact_ids=(fact_id,),
        lineage_fact_ids=(fact_id,),
        new_support_fact_ids=(fact_id,),
        removed_support_fact_ids=(),
        last_supported_at=NOW,
    )

    assert not await normal.persist_mutations(existing_cross_fish_batch, ())
    assert not await normal.persist_mutations(
        existing_cross_fish_batch,
        (mutation,),
    )
    unchanged = await postgres_pool.fetchrow(
        """
        SELECT version, through_fact_sequence
        FROM memory_snapshots
        WHERE id = $1
        """,
        header["id"],
    )
    assert unchanged is not None
    assert dict(unchanged) == {"version": 1, "through_fact_sequence": 0}
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_revisions WHERE snapshot_id = $1",
            header["id"],
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_items WHERE snapshot_id = $1",
            header["id"],
        )
        == 0
    )
