from datetime import timedelta

import pytest

from fish_memory.experimental.materials import MATERIAL_OBSERVATION, MaterialStore
from fish_memory.history.models import render_history
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.observations.store import ObservationStore
from tests.integration.test_session_history import START, message, setup


@pytest.mark.asyncio
async def test_material_reaches_history_without_waking_a_decision(postgres_pool):
    fish, session, history = await setup(postgres_pool)
    await history.flush(fish, session)
    notifications = []
    history.on_unit = lambda fish, session, unit, at: notifications.append(unit)
    text = "想起：用户去年爬过华山（2天前记下的）"
    await ObservationStore(postgres_pool).record_event(
        fish,
        ImportantObservationEvent(
            source="experimental",
            observation_type=MATERIAL_OBSERVATION,
            semantic_key="recall:test",
            summary=text,
            occurred_at=START + timedelta(minutes=1),
            payload={"session_id": str(session), "context_text": text},
        ),
    )
    await history.flush(fish, session)
    [unit] = notifications
    assert unit.label == "event" and unit.content == text
    # It reaches speech through TA's thoughts, not as something just said.
    assert not unit.triggers_decision
    assert not unit.interrupts_observation
    assert "event: " + text in render_history(await history.read(fish, session))
    store = MaterialStore(postgres_pool)
    assert await store.surfaced(fish, START) == {"recall:test"}


@pytest.mark.asyncio
async def test_material_reads_questions_and_cut_off_lines(postgres_pool):
    pool = postgres_pool
    fish, session, _ = await setup(pool)
    asked = await message(
        pool, fish, session, "你周末想去哪座山呀？", START, role="assistant"
    )
    await message(
        pool,
        fish,
        session,
        "我先去看看天气。",
        START + timedelta(seconds=20),
        role="assistant",
    )
    cut = await message(
        pool,
        fish,
        session,
        "我给你讲讲那个故事",
        START + timedelta(minutes=3),
        role="assistant",
    )
    await pool.execute("UPDATE messages SET status='interrupted' WHERE id=$1", cut)
    await pool.execute(
        "INSERT INTO message_unsaid(message_id,content) VALUES($1,$2)",
        cut,
        "，后来小猫找到了家。",
    )
    store = MaterialStore(pool)
    [row] = await store.asked(fish, session, START, START + timedelta(minutes=1))
    assert row["id"] == asked
    assert row["next_at"] == START + timedelta(seconds=20)
    [row] = await store.interrupted(fish, session, START, START + timedelta(minutes=5))
    assert (row["id"], row["content"], row["unsaid"]) == (
        cut,
        "我给你讲讲那个故事",
        "，后来小猫找到了家。",
    )
