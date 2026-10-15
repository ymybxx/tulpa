from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import pytest

from fish_memory.answer.context import INTERRUPTION_MARKER
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.extensions.models import EventDefinition, Instance
from fish_memory.extensions.store import ExtensionStore
from fish_memory.history.models import render_history
from fish_memory.history.service import SessionHistoryService
from fish_memory.history.store import SessionHistoryStore
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.observations.store import ObservationStore

START = datetime(2026, 9, 18, 8, tzinfo=UTC)


class Summarizer:
    def __init__(self):
        self.inputs = []
        self.fail = False

    async def stream(self, messages, **kwargs):
        self.inputs.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("model unavailable")
        yield ChatStreamChunk(content="用户和宠物讨论了电影，随后一起看了一段预告片。")


async def setup(pool):
    await run_migrations(cast(MigrationPool, pool))
    fish, session = uuid4(), uuid4()
    await pool.execute("INSERT INTO fishes(id) VALUES($1)", fish)
    await pool.execute(
        "INSERT INTO fish_sessions(id,fish_id,created_at) VALUES($1,$2,$3)",
        session,
        fish,
        START,
    )
    return fish, session, SessionHistoryStore(pool)


@pytest.mark.asyncio
async def test_finished_result_enters_latest_chunk_and_notifies_once(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "回忆一下去年那次旅行", START)
    completed = START + timedelta(minutes=21)
    observation = await ObservationStore(pool).record_event(
        fish,
        ImportantObservationEvent(
            source="experimental",
            observation_type="experimental.result",
            semantic_key=str(uuid4()),
            summary="回忆结果",
            occurred_at=completed,
            payload={
                "session_id": str(session),
                "context_text": "关于旅行的回忆结果出来了：去年去了杭州。",
            },
        ),
    )
    observed = []
    store.on_unit = lambda fish, session, unit, received_at: observed.append(unit)
    await store.flush(fish, session)
    result = next(unit for unit in observed if unit.source_id == observation.id)
    assert result.started_at == result.ended_at == completed
    chunks = await store.read(fish, session)
    assert result in chunks[-1].units
    assert len(chunks) == 2
    observed.clear()
    await store.flush(fish, session)
    assert not observed


@pytest.mark.asyncio
async def test_started_work_enters_flat_history_without_decision_trigger(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    observations = ObservationStore(pool)
    origin = uuid4()
    texts = [
        "关于“演员是谁”的搜索已开始，正在等待结果。",
        "关于“演员是谁”的搜索结果出来了。\n结果如下：已找到演员信息。",
    ]
    for index, (kind, text) in enumerate(
        zip(("started", "result"), texts, strict=True)
    ):
        await observations.record_event(
            fish,
            ImportantObservationEvent(
                source="experimental",
                observation_type=f"experimental.{kind}",
                semantic_key=f"{origin}:{kind}",
                summary=text,
                occurred_at=START + timedelta(seconds=index * 10),
                payload={
                    "session_id": str(session),
                    "origin_id": str(origin),
                    "context_text": text,
                },
            ),
        )
    notifications = []
    store.on_unit = lambda fish, session, unit, received_at: notifications.append(unit)
    await store.flush(fish, session)
    assert [unit.triggers_decision for unit in notifications] == [False, True]
    assert [unit.reports_work for unit in notifications] == [False, True]
    history = render_history(await store.read(fish, session))
    assert history.count("event:") == 2
    assert history.index("搜索已开始") < history.index("搜索结果出来了")
    assert "origin_id" not in history and "triggers_decision" not in history
    notifications.clear()
    await store.flush(fish, session)
    assert not notifications


@pytest.mark.asyncio
async def test_host_events_are_idempotent_history_inputs_without_assistant_message(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    events = ExtensionStore(pool)
    identifier = uuid4()
    for _ in range(2):
        await events.receive_host_event(
            fish,
            session,
            identifier,
            "codex.completed",
            {
                "title": "测试任务",
                "result": "完成三个检查。",
            },
        )
    await store.flush(fish, session)
    text = render_history(await store.read(fish, session))
    assert text.count("event: codex.completed") == 1
    assert "完成三个检查。" in text
    assert (
        await pool.fetchval("SELECT count(*) FROM messages WHERE fish_id=$1", fish) == 0
    )


@pytest.mark.asyncio
async def test_sent_in_events_wake_nobody_and_say_whether_they_are_for_the_user(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    events = ExtensionStore(pool)
    notifications = []
    store.on_unit = lambda fish, session, unit, received_at: notifications.append(unit)
    await events.receive_host_event(
        fish,
        session,
        uuid4(),
        "external.message",
        {"text": "今晚十点有雨"},
        wakes=False,
    )
    await events.receive_host_event(
        fish,
        session,
        uuid4(),
        "external.message",
        {"text": "快递到了"},
        wakes=False,
        reports_work=True,
    )
    await events.receive_host_event(
        fish, session, uuid4(), "codex.completed", {"result": "完成"}
    )
    await store.flush(fish, session)
    assert [unit.triggers_decision for unit in notifications] == [False, False, True]
    assert [unit.reports_work for unit in notifications] == [False, True, False]
    text = render_history(await store.read(fish, session))
    assert "今晚十点有雨" in text and "快递到了" in text
    assert "wakes" not in text and "reports_work" not in text


@pytest.mark.asyncio
async def test_plugin_narratives_follow_thought_context_text_path(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    events = ExtensionStore(pool)
    instance = Instance(uuid4(), "test.live", fish, session, False, "test-token")
    definitions = [
        ("live.reply", "直播间观众「${viewer_name}」发来弹幕：${text}"),
        (
            "live.gift",
            "直播间观众「${viewer_name}」送出 ${gift_count} 个「${gift_name}」。",
        ),
        ("live.enter", "直播间观众「${viewer_name}」进入了直播间。"),
    ]
    raw = {
        "viewer_id": "platform-viewer-123",
        "viewer_name": "阿月",
        "text": "刚才讲的那个游戏叫什么？",
        "gift_count": 3,
        "gift_name": "小心心",
    }
    notifications = []
    store.on_unit = lambda fish, session, unit, received_at: notifications.append(unit)
    for event_type, template in definitions:
        definition = EventDefinition(
            type=event_type,
            schema={
                "type": "object",
                "properties": {
                    key: {"type": "integer" if isinstance(value, int) else "string"}
                    for key, value in raw.items()
                },
                "x-fish-history-template": template,
            },
        )
        await events.definition(instance, definition)
        identifier = uuid4()
        first, created = await events.receive(
            instance, identifier, definition, raw, "stored_only", None
        )
        assert created and first["data"] == raw
        repeated, created = await events.receive(
            instance, identifier, definition, raw, "stored_only", None
        )
        assert not created and repeated == first
    await store.flush(fish, session)
    text = render_history(await store.read(fish, session))
    assert [line for line in text.splitlines() if line.startswith("event:")] == [
        "event: 直播间观众「阿月」发来弹幕：刚才讲的那个游戏叫什么？",
        "event: 直播间观众「阿月」送出 3 个「小心心」。",
        "event: 直播间观众「阿月」进入了直播间。",
    ]
    assert "viewer_id" not in text and "platform-viewer-123" not in text
    assert "{" not in text and "live." not in text
    assert len(notifications) == 3
    assert all(unit.triggers_decision for unit in notifications)


async def message(pool, fish, session, text, at, *, role="user", offset=None):
    identifier = uuid4()
    await pool.execute(
        """INSERT INTO messages
           (id,fish_id,session_id,role,content,status,created_at,consumed_offset)
           VALUES($1,$2,$3,$4,$5,$8,$6,$7)""",
        identifier,
        fish,
        session,
        role,
        text,
        at,
        offset,
        "initialized" if role == "user" else "completed",
    )
    return identifier


@pytest.mark.asyncio
async def test_flat_timeline_late_events_and_played_content(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "先看预告片", START + timedelta(seconds=1))
    assistant = await message(
        pool,
        fish,
        session,
        "好，尚未播放",
        START + timedelta(seconds=3),
        role="assistant",
        offset=1,
    )
    await store.flush(fish, session)
    event = uuid4()
    await pool.execute(
        """INSERT INTO observations
           (id,fish_id,kind,source,observation_type,semantic_key,
            summary,payload,started_at,ended_at)
           VALUES($1,$2,'event','desktop','app.changed','editor','打开编辑器','{}',$3,$3)""",
        event,
        fish,
        START + timedelta(seconds=2),
    )
    await store.flush(fish, session)
    chunks = await store.read(fish, session)
    text = render_history(chunks)
    assert (
        text.index("u: 先看预告片")
        < text.index("event: 打开编辑器")
        < text.index("a: 好")
    )
    assert "尚未播放" not in text and "dialogue" not in text
    assert INTERRUPTION_MARKER in text
    assert len(chunks) == 1 and chunks[0].unit_count == 3
    await store.flush(fish, session)
    assert await store.read(fish, session) == chunks
    await pool.execute("UPDATE messages SET consumed_offset=3 WHERE id=$1", assistant)
    await store.flush(fish, session)
    assert (await store.read(fish, session))[0].unit_count == 3


@pytest.mark.asyncio
async def test_recursive_compaction_and_recent_tail_survive_restart(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    for index in range(3):
        await message(
            pool, fish, session, "电影" * 220, START + timedelta(minutes=10 * index)
        )
    await store.flush(fish, session)
    llm = Summarizer()
    service = SessionHistoryService(
        store, llm, now=lambda: START + timedelta(hours=1), token_limit=1_000
    )
    await service.compact(fish, session)
    chunks = await store.read(fish, session)
    assert len(chunks) == 2 and chunks[0].summary and not chunks[1].summary
    assert chunks[0].ended_at == START + timedelta(minutes=20)
    for index in (3, 4):
        await message(
            pool, fish, session, "新话题" * 160, START + timedelta(minutes=index * 10)
        )
    await store.flush(fish, session)
    await service.compact(fish, session)
    assert "summary: 用户和宠物" in llm.inputs[-1]
    restored = SessionHistoryStore(pool)
    await restored.flush(fish, session)
    assert await restored.read(fish, session) == await store.read(fish, session)
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM messages WHERE session_id=$1", session
        )
        == 5
    )


@pytest.mark.asyncio
async def test_expiration_uses_chunk_end_and_oversized_chunk_compacts_itself(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "内容" * 800, START)
    await store.flush(fish, session)
    llm = Summarizer()
    service = SessionHistoryService(
        store, llm, now=lambda: START + timedelta(hours=5, minutes=5), token_limit=1_000
    )
    await service.compact(fish, session)
    chunks = await store.read(fish, session)
    assert len(chunks) == 1 and chunks[0].summary
    assert chunks[0].started_at == START
    assert chunks[0].ended_at == START + timedelta(minutes=10)
    await store.expire(fish, session, START + timedelta(minutes=11))
    assert not await store.read(fish, session)
    await store.flush(fish, session)
    assert not await store.read(fish, session)
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM messages WHERE session_id=$1", session
        )
        == 1
    )


@pytest.mark.asyncio
async def test_failed_or_stale_compaction_keeps_source_changes(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    original = await message(pool, fish, session, "旧内容" * 400, START)
    await store.flush(fish, session)
    selected = await store.read(fish, session)
    await message(pool, fish, session, "插入的消息", START + timedelta(seconds=2))
    await pool.execute(
        "UPDATE messages SET content=content || '修正' WHERE id=$1", original
    )
    await store.flush(fish, session)
    assert not await store.replace_prefix(fish, session, selected, "旧摘要")
    llm = Summarizer()
    llm.fail = True
    service = SessionHistoryService(store, llm, now=lambda: START, token_limit=1_000)
    await service.compact(fish, session)
    assert "插入的消息" in render_history(await store.read(fish, session))
    assert not (await store.read(fish, session))[0].summary


@pytest.mark.asyncio
async def test_sessions_are_isolated_and_source_deletion_removes_current_unit(
    postgres_pool,
):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    other = uuid4()
    await pool.execute(
        "INSERT INTO fish_sessions(id,fish_id,created_at) VALUES($1,$2,$3)",
        other,
        fish,
        START,
    )
    first = await message(pool, fish, session, "甲会话", START)
    await message(pool, fish, other, "乙会话", START)
    await store.flush(fish, session)
    await store.flush(fish, other)
    assert "乙会话" not in render_history(await store.read(fish, session))
    await pool.execute("DELETE FROM messages WHERE id=$1", first)
    await store.flush(fish, session)
    assert not await store.read(fish, session)
    with pytest.raises(LookupError):
        await store.flush(UUID(int=1), other)


@pytest.mark.asyncio
async def test_append_during_compaction_is_kept_as_raw_content(postgres_pool):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "较早的讨论" * 250, START)
    await store.flush(fish, session)
    selected = await store.read(fish, session)
    await message(
        pool, fish, session, "压缩期间新来的消息", START + timedelta(seconds=5)
    )
    await store.flush(fish, session)
    assert await store.replace_prefix(fish, session, selected, "先前讨论了电影。")
    chunks = await store.read(fish, session)
    text = render_history(chunks)
    assert "summary: 先前讨论了电影" in text
    assert "u: 压缩期间新来的消息" in text
    assert "较早的讨论" not in text
    assert chunks[0].unit_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [5, 15])
async def test_insert_after_summary_sources_preserves_summary(postgres_pool, seconds):
    pool = postgres_pool
    fish, session, store = await setup(pool)
    await message(pool, fish, session, "较早的讨论", START)
    await store.flush(fish, session)
    assert await store.replace_prefix(
        fish, session, await store.read(fish, session), "先前讨论了电影。"
    )
    await message(pool, fish, session, "未压缩的后续", START + timedelta(seconds=10))
    await store.flush(fish, session)
    await message(pool, fish, session, "新记录", START + timedelta(seconds=seconds))
    await store.flush(fish, session)
    chunks = await store.read(fish, session)
    text = render_history(chunks)
    assert "summary: 先前讨论了电影。" in text and "较早的讨论" not in text
    assert text.index("summary:") < min(text.index("新记录"), text.index("未压缩"))
    assert (text.index("新记录") < text.index("未压缩的后续")) is (seconds < 10)
    assert chunks[0].unit_count == 3


@pytest.mark.asyncio
async def test_empty_media_and_direct_computer_events_share_the_session_history(
    postgres_pool,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx
    from fastapi import FastAPI

    from fish_memory.history.routes import history_router
    from fish_memory.observations.environment_history_models import (
        SemanticDocument,
        SemanticSnapshot,
    )
    from fish_memory.observations.environment_history_store import (
        EnvironmentHistoryStore,
    )

    pool = postgres_pool
    fish, session, store = await setup(pool)
    history = SessionHistoryService(
        store, Summarizer(), now=lambda: START + timedelta(minutes=1)
    )
    app = FastAPI()
    app.include_router(history_router)
    app.state.session_history = history
    app.state.fish_store = SimpleNamespace(exists=AsyncMock(return_value=True))
    event = {
        "schemaVersion": 1,
        "ordinal": 1,
        "id": str(uuid4()),
        "timestamp": (START + timedelta(seconds=1)).isoformat(),
        "kind": "window.changed",
        "app": {"name": "编辑器"},
        "window": {"title": "项目代码"},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        for _ in range(2):
            response = await client.post(
                f"/internal/fishes/{fish}/history/events", json={"events": [event]}
            )
            assert response.status_code == 200, response.text
    media = EnvironmentHistoryStore(pool)
    await media.save(
        SemanticSnapshot(
            id=uuid4(),
            fish_id=fish,
            session_id=session,
            capture_session_id=uuid4(),
            window_index=0,
            period_seconds=5,
            started_at=START,
            ended_at=START + timedelta(seconds=5),
            document=SemanticDocument(summary_version=4),
            input_digest="0" * 64,
            created_at=START,
        )
    )
    await store.flush(fish, session)
    chunks = await store.read(fish, session)
    assert sum(chunk.unit_count for chunk in chunks) == 1
    assert "event: window.changed" in render_history(chunks)
    assert "项目代码" in render_history(chunks)
    assert "已处理的影音时间段" not in render_history(chunks)
    await history.close()
