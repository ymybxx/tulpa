from datetime import timedelta
from uuid import uuid4

import pytest

from fish_memory.answer.history_messages import history_role_messages
from fish_memory.history.models import render_history
from fish_memory.history.store import SessionHistoryStore
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.observations.store import ObservationStore
from tests.integration.test_session_history import START, message, setup


async def add_thought(pool, fish, session, seconds, **payload):
    started_at = START + timedelta(seconds=seconds)
    event = ImportantObservationEvent(
        source="experimental",
        observation_type="experimental.thought",
        semantic_key=str(uuid4()),
        summary="宠物的内心想法（尚未表达）：这段旋律和刚才有些不同。",
        occurred_at=started_at,
        payload={
            "session_id": str(session),
            "thought_started_at": started_at.isoformat(),
            **payload,
        },
    )
    return await ObservationStore(pool).record_event(fish, event)


@pytest.mark.asyncio
async def test_gate_counts_only_session_thoughts_after_confirmed_speech(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    now = START + timedelta(seconds=100)

    async def eligible():
        return await store.has_thought_since_last_speech(fish, session, now)

    assert not await eligible()
    other = uuid4()
    await pool.execute(
        "INSERT INTO fish_sessions(id,fish_id) VALUES($1,$2)", other, fish
    )
    await add_thought(pool, fish, other, 10)
    assert not await eligible()
    await add_thought(pool, fish, session, 10)
    assert await eligible()  # The first utterance also requires a thought.

    await message(pool, fish, session, "用户说话", START + timedelta(seconds=20))
    unplayed = await message(
        pool,
        fish,
        session,
        "尚未播放的回复",
        START + timedelta(seconds=30),
        role="assistant",
        offset=0,
    )
    assert await eligible()
    await pool.execute(
        "UPDATE messages SET consumed_offset=2,"
        "fact_extraction_status='waiting_consumption' WHERE id=$1",
        unplayed,
    )
    assert await eligible()  # Not yet acknowledged into spoken History.
    await pool.execute(
        "UPDATE messages SET status='interrupted',fact_extraction_status='pending' "
        "WHERE id=$1",
        unplayed,
    )
    assert not await eligible()

    await add_thought(pool, fish, session, 40, include_in_context=False)
    await add_thought(pool, fish, session, 101)  # Future timestamps do not qualify.
    result = ImportantObservationEvent(
        source="experimental",
        observation_type="experimental.result",
        semantic_key=str(uuid4()),
        summary="回忆结果出来了。",
        occurred_at=START + timedelta(seconds=45),
        payload={"session_id": str(session)},
    )
    await ObservationStore(pool).record_event(fish, result)
    assert not await eligible()
    await add_thought(pool, fish, session, 50)
    assert await eligible()

    silent = await message(
        pool,
        fish,
        session,
        "没有播出",
        START + timedelta(seconds=60),
        role="assistant",
        offset=0,
    )
    await pool.execute("UPDATE messages SET status='interrupted' WHERE id=$1", silent)
    assert await eligible()
    await message(
        pool,
        fish,
        session,
        "接着说了新的内容",
        START + timedelta(seconds=70),
        role="assistant",
        offset=2,
    )
    assert not await eligible()


@pytest.mark.asyncio
async def test_gate_survives_projection_compaction_and_store_recreation(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    now = START + timedelta(minutes=1)
    await message(pool, fish, session, "刚才的发言", START, role="assistant", offset=-1)
    await store.flush(fish, session)
    thought = await add_thought(pool, fish, session, 5)
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM session_history_units WHERE source_id=$1)",
        thought.id,
    )
    assert await store.has_thought_since_last_speech(fish, session, now)
    await store.flush(fish, session)
    chunks = await store.read(fish, session)
    assert await store.replace_prefix(
        fish, session, chunks, "聊过音乐，随后宠物产生了联想。"
    )
    restored = SessionHistoryStore(pool)
    for _ in range(2):
        assert await restored.has_thought_since_last_speech(fish, session, now)
    await message(
        pool,
        fish,
        session,
        "新的发言",
        START + timedelta(seconds=10),
        role="assistant",
        offset=-1,
    )
    # A source write is authoritative even before the next projection flush.
    assert not await restored.has_thought_since_last_speech(fish, session, now)


@pytest.mark.asyncio
@pytest.mark.parametrize("thought_at,expected", [(9, False), (10, False), (11, True)])
async def test_gate_uses_thought_start_even_when_it_finishes_after_speech(
    postgres_pool,
    thought_at,
    expected,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(
        pool,
        fish,
        session,
        "刚才已经说过",
        START + timedelta(seconds=10),
        role="assistant",
        offset=-1,
    )
    # A late-arriving old thought must not be mistaken for a fresh one.
    await add_thought(
        pool,
        fish,
        session,
        thought_at,
        thought_completed_at=(START + timedelta(seconds=15)).isoformat(),
    )
    assert (
        await store.has_thought_since_last_speech(
            fish, session, START + timedelta(seconds=20)
        )
        is expected
    )


@pytest.mark.asyncio
async def test_legacy_completion_timestamp_does_not_establish_thought_origin(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "刚才说过", START, role="assistant", offset=-1)
    legacy = await add_thought(pool, fish, session, 5, thought_started_at=None)
    await pool.execute(
        "UPDATE observations SET payload=payload-'thought_started_at' WHERE id=$1",
        legacy.id,
    )
    assert not await store.has_thought_since_last_speech(
        fish, session, START + timedelta(seconds=10)
    )
    await store.flush(fish, session)
    assert any(
        u.source_id == legacy.id
        for c in await store.read(fish, session)
        for u in c.units
    )  # Preserve old History without inventing missing start times.
    await add_thought(pool, fish, session, 6)
    assert await store.has_thought_since_last_speech(
        fish, session, START + timedelta(seconds=10)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("compacted", [False, True])
async def test_delayed_thought_returns_to_origin_before_later_speech(
    postgres_pool, compacted
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "用户刚放了首歌", START + timedelta(seconds=598))
    await message(
        pool,
        fish,
        session,
        "宠物已经接着说了话",
        START + timedelta(seconds=601),
        role="assistant",
        offset=-1,
    )
    await store.flush(fish, session)
    if compacted:
        assert await store.replace_prefix(
            fish, session, await store.read(fish, session), "用户放歌，宠物随后发言。"
        )
    observed = []
    store.on_unit = lambda fish, session, unit, received_at: observed.append(unit)
    thought = await add_thought(
        pool,
        fish,
        session,
        599,
        thought_completed_at=(START + timedelta(seconds=605)).isoformat(),
    )
    # Completion crosses a ten-minute boundary, but the origin stays before it.
    assert await store.flush(fish, session) == ()
    assert len(observed) == 1 and observed[0].source_id == thought.id
    assert not observed[0].triggers_decision
    assert not observed[0].interrupts_observation
    assert (
        observed[0].started_at == observed[0].ended_at == START + timedelta(seconds=599)
    )
    chunks = await store.read(fish, session)
    assert not any(c.summary for c in chunks)
    assert sum(c.unit_count for c in chunks) == 3
    origin = next(c for c in chunks if observed[0] in c.units)
    assert origin.started_at <= START + timedelta(seconds=599) < origin.ended_at
    for text in (
        render_history(chunks),
        "\n".join(m["content"] for m in history_role_messages(chunks)),
    ):
        assert (
            text.index("用户刚放了首歌")
            < text.index("宠物的内心想法")
            < text.index("宠物已经接着说了话")
        )
        assert "thought_completed_at" not in text
    assert not await store.has_thought_since_last_speech(
        fish, session, START + timedelta(seconds=610)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_seconds", [4, 604])
async def test_late_thought_invalidates_compaction_that_would_reorder_it(
    postgres_pool, speech_seconds
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "用户最初的话", START)
    await message(
        pool,
        fish,
        session,
        "宠物后来的话",
        START + timedelta(seconds=speech_seconds),
        role="assistant",
        offset=-1,
    )
    await store.flush(fish, session)
    selected = await store.read(fish, session)
    await add_thought(pool, fish, session, 2)
    await store.flush(fish, session)
    assert not await store.replace_prefix(fish, session, selected, "缺少思考的旧摘要")
    text = render_history(await store.read(fish, session))
    assert (
        text.index("用户最初的话")
        < text.index("宠物的内心想法")
        < text.index("宠物后来的话")
    )
