import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from pydantic import SecretStr

from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.db import migrations
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.service import ExperimentalService
from fish_memory.experimental.thoughts import ThoughtWorker
from fish_memory.history.models import render_history
from fish_memory.history.service import SessionHistoryService
from fish_memory.observations.environment_history_models import (
    SemanticDocument,
    SemanticSnapshot,
)
from fish_memory.observations.environment_history_store import EnvironmentHistoryStore
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.observations.store import ObservationContextGuard, ObservationStore
from tests.integration.test_session_history import START, message, setup
from tests.test_experimental_thoughts import THOUGHTS, Reasoning


def thought(session, text="我想继续听听这段旋律。"):
    content = f"宠物的内心想法（尚未表达）：{text}"
    return ImportantObservationEvent(
        source="experimental",
        observation_type="experimental.thought",
        semantic_key=f"thought:{uuid4()}",
        summary=content,
        occurred_at=START + timedelta(minutes=1),
        payload={"session_id": str(session), "context_text": content},
    )


async def audiovisual(pool, fish, session, document):
    return await EnvironmentHistoryStore(pool).save(
        SemanticSnapshot(
            id=uuid4(),
            fish_id=fish,
            capture_session_id=uuid4(),
            window_index=0,
            period_seconds=5,
            started_at=START,
            ended_at=START + timedelta(seconds=5),
            document=document,
            input_digest="test-window",
            created_at=START,
            session_id=session,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unchanged", [False, True])
async def test_hidden_av_receipts_are_durable_without_context_progress(
    postgres_pool,
    unchanged,
):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    await history.flush(fish, session)
    before = await history.context_revision(fish, session)
    document = SemanticDocument(
        summary_version=4, unchanged_from=uuid4() if unchanged else None
    )
    receipt = await audiovisual(pool, fish, session, document)
    assert await EnvironmentHistoryStore(pool).get(fish, receipt.id) == receipt
    assert await history.context_revision(fish, session) == before
    assert not await history.pending_sessions()

    # Initial History materialization must not re-introduce hidden receipts.
    await pool.execute(
        "UPDATE session_history_contexts SET initialized=false WHERE session_id=$1",
        session,
    )
    await history.flush(fish, session)
    assert not await history.read(fish, session)
    assert await history.context_revision(fish, session) == before


@pytest.mark.asyncio
async def test_observation_visibility_transitions_still_invalidate_context(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    await history.flush(fish, session)
    before = await history.context_revision(fish, session)
    event = thought(session)
    event = replace(event, payload={**event.payload, "include_in_context": False})
    saved = await ObservationStore(pool).record_event(fish, event)
    assert await history.context_revision(fish, session) == before
    for visible in (True, False):
        await pool.execute(
            "UPDATE observations SET payload=jsonb_set(payload,"
            "'{include_in_context}',$2::jsonb) WHERE id=$1",
            saved.id,
            json.dumps(visible),
        )
        after = await history.context_revision(fish, session)
        assert after > before  # Advance before projection, including removal.
        before = after
        await history.flush(fish, session)
        text = render_history(await history.read(fish, session))
        assert (event.summary in text) is visible
    await pool.execute(
        "UPDATE observations SET summary='仍不进入上下文' WHERE id=$1", saved.id
    )
    await pool.execute("DELETE FROM observations WHERE id=$1", saved.id)
    assert await history.context_revision(fish, session) == before


@pytest.mark.asyncio
async def test_visibility_migration_upgrades_existing_context_fence(
    postgres_pool, monkeypatch
):
    pool = postgres_pool
    with monkeypatch.context() as patch:
        patch.setattr(
            migrations,
            "MIGRATIONS",
            tuple(m for m in migrations.MIGRATIONS if m.version <= 46),
        )
        fish, session, history = await setup(pool)
    await history.flush(fish, session)
    before = await history.context_revision(fish, session)
    await audiovisual(pool, fish, session, SemanticDocument(summary_version=4))
    # Reproduce the shipped bug: an invisible receipt advances the fence.
    assert await history.context_revision(fish, session) > before
    await history.flush(fish, session)
    assert not await history.read(fish, session)
    await run_migrations(cast(MigrationPool, pool))
    before = await history.context_revision(fish, session)
    await audiovisual(pool, fish, session, SemanticDocument(summary_version=4))
    assert await history.context_revision(fish, session) == before


@pytest.mark.asyncio
async def test_unprojected_source_changes_invalidate_thoughts(postgres_pool):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    await history.flush(fish, session)
    observations = ObservationStore(pool)
    before = await history.context_revision(fish, session)
    identifier = await message(pool, fish, session, "换个话题", START)
    assert not await history.read(fish, session)  # Still waiting in the inbox.
    assert await history.context_revision(fish, session) > before
    assert (
        await observations.record_event(
            fish,
            thought(session),
            guard=ObservationContextGuard(session, before, lambda: True),
        )
        is None
    )
    await history.flush(fish, session)

    # An edit and a deletion must invalidate even though neither appends a message.
    for sql in (
        "UPDATE messages SET content='先安静一会儿' WHERE id=$1",
        "DELETE FROM messages WHERE id=$1",
    ):
        before = await history.context_revision(fish, session)
        await pool.execute(sql, identifier)
        assert (
            await observations.record_event(
                fish,
                thought(session),
                guard=ObservationContextGuard(session, before, lambda: True),
            )
            is None
        )
        await history.flush(fish, session)
    assert (
        await pool.fetchval("SELECT count(*) FROM observations WHERE fish_id=$1", fish)
        == 0
    )


@pytest.mark.asyncio
async def test_thought_is_visible_once_without_behavior_or_audiovisual_boundary(
    postgres_pool,
):
    fish, session, history = await setup(postgres_pool)
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    notifications = []
    history.on_unit = lambda fish, session, unit, at: notifications.append(unit)
    event = thought(session)
    recorded = await ObservationStore(postgres_pool).record_event(
        fish, event, guard=ObservationContextGuard(session, version, lambda: True)
    )
    assert recorded is not None
    assert await history.context_revision(fish, session) > version
    assert await history.flush(fish, session) == ()
    assert len(notifications) == 1
    assert not notifications[0].triggers_decision
    assert not notifications[0].interrupts_observation
    chunks = await history.read(fish, session)
    assert chunks[0].unit_count == 1
    assert render_history(chunks).count(event.summary) == 1
    assert "event: 宠物的内心想法（尚未表达）" in render_history(chunks)
    await history.flush(fish, session)
    assert len(notifications) == 1


@pytest.mark.asyncio
async def test_guard_waits_for_concurrent_source_commit_and_rejects_old_version(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    identifier = await message(pool, fish, session, "原始话题", START)
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    pending = None
    try:
        async with pool.acquire() as source, source.transaction():
            await source.execute(
                "UPDATE messages SET content='新话题' WHERE id=$1", identifier
            )
            pending = asyncio.create_task(
                ObservationStore(pool).record_event(
                    fish,
                    thought(session),
                    guard=ObservationContextGuard(session, version, lambda: True),
                )
            )
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(pending), 0.05)
        assert await asyncio.wait_for(pending, 3) is None
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM observations WHERE fish_id=$1", fish
            )
            == 0
        )
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_lifecycle_change_during_insert_rolls_back_event_and_version(
    postgres_pool,
):
    fish, session, history = await setup(postgres_pool)
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    checks = iter((True, False))
    recorded = await ObservationStore(postgres_pool).record_event(
        fish,
        thought(session),
        guard=ObservationContextGuard(session, version, lambda: next(checks)),
    )
    assert recorded is None
    assert await history.context_revision(fish, session) == version
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM observations WHERE fish_id=$1", fish
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM session_history_pending WHERE session_id=$1", session
        )
        == 0
    )


@pytest.mark.asyncio
async def test_profile_and_static_snapshot_changes_advance_same_context_fence(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    await pool.execute("UPDATE fishes SET setting='喜欢音乐' WHERE id=$1", fish)
    assert await history.context_revision(fish, session) > version
    version = await history.context_revision(fish, session)
    snapshot = await pool.fetchval(
        "INSERT INTO memory_snapshots"
        "(fish_id,snapshot_type,content,through_fact_sequence,version) "
        "VALUES($1,'static','{}',0,1) RETURNING id",
        fish,
    )
    assert await history.context_revision(fish, session) > version
    version = await history.context_revision(fish, session)
    await pool.execute(
        "INSERT INTO memory_snapshot_items"
        "(snapshot_id,summary,support_fact_ids,lineage_fact_ids,last_supported_at) "
        "VALUES($1,'用户喜欢音乐',$2::uuid[],$2::uuid[],$3)",
        snapshot,
        [uuid4()],
        START,
    )
    assert await history.context_revision(fish, session) > version
    version = await history.context_revision(fish, session)
    await pool.execute(
        "INSERT INTO memory_snapshots"
        "(fish_id,session_id,snapshot_type,content,through_fact_sequence,version) "
        "VALUES($1,$2,'dynamic','{}',0,1)",
        fish,
        session,
    )
    assert await history.context_revision(fish, session) == version


@pytest.mark.asyncio
async def test_compaction_expiration_and_playback_correction_invalidate_old_context(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    identifier = await message(
        pool, fish, session, "已经说完，还未说完。", START, role="assistant", offset=4
    )
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    await pool.execute(
        "UPDATE messages SET consumed_offset=3,status='interrupted' WHERE id=$1",
        identifier,
    )
    assert await history.context_revision(fish, session) > version
    await history.flush(fish, session)
    chunks = await history.read(fish, session)
    version = await history.context_revision(fish, session)
    assert await history.replace_prefix(fish, session, chunks, "宠物的发言被打断。")
    assert await history.context_revision(fish, session) > version
    version = await history.context_revision(fish, session)
    await history.expire(fish, session, START + timedelta(hours=1))
    assert await history.context_revision(fish, session) > version


@pytest.mark.asyncio
async def test_other_session_progress_does_not_expire_current_thought(postgres_pool):
    pool = postgres_pool
    fish, session, history = await setup(pool)
    await history.flush(fish, session)
    version = await history.context_revision(fish, session)
    other = uuid4()
    await pool.execute(
        "INSERT INTO fish_sessions(id,fish_id) VALUES($1,$2)", other, fish
    )
    await message(pool, fish, other, "另一个会话", START)
    assert await history.context_revision(fish, session) == version
    assert (
        await ObservationStore(pool).record_event(
            fish,
            thought(session),
            guard=ObservationContextGuard(session, version, lambda: True),
        )
        is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["new", "repeat", "empty_av", "changed_av"])
async def test_next_thought_round_reads_selected_event_through_real_context_loader(
    postgres_pool,
    mode,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "一起听歌吧", START)
    await store.flush(fish, session)
    reasoning = Reasoning()
    history = SessionHistoryService(store, reasoning, now=lambda: START)
    contexts = AnswerContextLoader(
        MessagesStore(pool),
        session_history=history,
        system_prompt="你喜欢音乐。",
        now=lambda: START,
    )
    states = []

    async def select(configuration, state, thoughts):
        states.append(state)
        if len(states) == 1:
            return 0
        if mode in {"empty_av", "changed_av"}:
            await audiovisual(
                pool,
                fish,
                session,
                SemanticDocument(
                    summary_version=4,
                    activity_summary="用户切换了网页。" if mode == "changed_av" else "",
                ),
            )
            return 1
        assert mode != "repeat", "a wholly repeated batch must not reach the judge"
        return 1

    async def close():
        pass

    service = ExperimentalService(
        ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only")),
        judge=SimpleNamespace(select_thought=select, close=close),
        chat=None,
        reasoning=reasoning,
        recall=None,
        observations=ObservationStore(pool),
        now=lambda: START,
        search_enabled=False,
        recall_token_budget=1000,
    )
    worker = ThoughtWorker(service, contexts, history, binding=lambda: (fish, session))
    try:
        assert await worker._run(fish, session) == "written"
        if mode == "repeat":
            # Candidate filtering now runs before the judge. Repeat the recorded
            # thought across the whole second batch, rather than selecting an
            # index that now refers to a different, fresh candidate.
            reasoning.raw = json.dumps(
                {
                    "thoughts": [
                        THOUGHTS[0] + suffix
                        for suffix in (
                            "我仍这样想。",
                            "我还是这想法。",
                            "我依然想到这点。",
                            "我仍在回想。",
                            "我还是想到这首歌。",
                        )
                    ]
                },
                ensure_ascii=False,
            )
        expected = {
            "new": "written",
            "repeat": "duplicate",
            "empty_av": "written",
            "changed_av": "stale",
        }[mode]
        assert await worker._run(fish, session) == expected
        assert THOUGHTS[0] not in states[0]
        second_input = "\n".join(
            message["content"] for message in reasoning.requests[1][0]
        )
        assert "宠物的内心想法（尚未表达）：" + THOUGHTS[0] in second_input
        assert all(candidate not in second_input for candidate in THOUGHTS[1:])
        assert len(states) == (1 if mode == "repeat" else 2)
        assert service.last_decision_at is None
        assert await pool.fetchval(
            "SELECT count(*) FROM observations WHERE fish_id=$1 "
            "AND observation_type='experimental.thought'",
            fish,
        ) == (2 if mode in {"new", "empty_av"} else 1)
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM messages "
                "WHERE session_id=$1 AND role='assistant'",
                session,
            )
            == 0
        )
    finally:
        await worker.close()
        await service.close()
        await history.close()
