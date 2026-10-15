from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fish_memory.clock import MutableClock
from fish_memory.db.migrations import run_migrations
from fish_memory.llm.log_routes import router
from fish_memory.llm.log_store import LlmLogStore
from fish_memory.llm.logs import CallJournal


@pytest.mark.asyncio
async def test_log_pagination_filter_detail_and_desktop_ingest(postgres_pool):
    await run_migrations(postgres_pool)
    store = LlmLogStore(postgres_pool)
    journal = CallJournal(store)
    app = FastAPI()
    app.state.llm_log_store, app.state.llm_call_journal = store, journal
    app.include_router(router)
    base = {
        "call_type": "plugin.select",
        "model": "fixture",
        "request_url": "https://model.invalid/v1/chat/completions",
        "status": "completed",
        "started_at": datetime.now(UTC),
        "ended_at": datetime.now(UTC),
        "duration_ms": 10,
        "http_status": 200,
        "error": None,
        "source": "host",
        "request": {"messages": [{"role": "user", "content": "first\nsecond"}]},
        "response": {"text": '{"a":"一\\n二"}'},
    }
    ids = [str(uuid4()) for _ in range(3)]
    try:
        for identifier in ids:
            await store.save({**base, "id": identifier})
        await store.save(
            {**base, "id": str(uuid4()), "call_type": "chat", "status": "failed"}
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = (
                await client.get(
                    "/internal/llm-logs",
                    params={"call_type": "plugin.select", "limit": 2},
                )
            ).json()
            assert [r["id"] for r in first["items"]] == list(reversed(ids[1:]))
            assert all(
                "request" not in row and "response" not in row for row in first["items"]
            )
            second = (
                await client.get(
                    "/internal/llm-logs",
                    params={
                        "call_type": "plugin.select",
                        "limit": 2,
                        "before": first["next_cursor"],
                    },
                )
            ).json()
            assert [r["id"] for r in second["items"]] == ids[:1]
            assert second["next_cursor"] is None
            detail = (await client.get(f"/internal/llm-logs/{ids[0]}")).json()
            assert (
                detail["request"] == base["request"]
                and detail["response"] == base["response"]
            )
            assert (
                await client.get("/internal/llm-logs", params={"limit": 1000})
            ).status_code == 422
            assert (await client.get("/internal/llm-logs/types")).json()["types"] == [
                "chat",
                "plugin.select",
            ]
            incoming = {
                **base,
                "id": str(uuid4()),
                "started_at": base["started_at"].isoformat(),
                "ended_at": base["ended_at"].isoformat(),
                "request_url": "https://user:secret@model.invalid/v1/chat/completions?key=hidden",
            }
            assert (
                await client.post("/internal/llm-logs", json=incoming)
            ).status_code == 200
            saved = await store.get(UUID(incoming["id"]))
            assert (
                saved["source"] == "desktop"
                and saved["request_url"] == base["request_url"]
            )
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_usage_counts_tokens_by_type_and_purpose(postgres_pool):
    from datetime import timedelta

    await run_migrations(postgres_pool)
    store = LlmLogStore(postgres_pool)
    app = FastAPI()
    app.state.llm_log_store = store
    app.include_router(router)
    now = datetime.now(UTC)
    day_start = now - timedelta(hours=3)

    def call(call_type, at, response, purpose=None):
        return {
            "id": str(uuid4()),
            "call_type": call_type,
            "model": "fixture",
            "request_url": "https://model.invalid/v1/chat/completions",
            "status": "completed",
            "started_at": at,
            "ended_at": at,
            "duration_ms": 10,
            "http_status": 200,
            "error": None,
            "source": "host",
            "request": {},
            "response": response,
            "purpose": purpose,
        }

    usage = {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    await store.save(
        call("experimental.thought.generate", now - timedelta(minutes=2), usage)
    )
    await store.save(
        call("experimental.thought.generate", now - timedelta(hours=2), usage)
    )
    await store.save(
        call("experimental.thought.generate", now - timedelta(days=2), usage)
    )
    await store.save(
        call("experimental.thought.generate", now - timedelta(days=9), usage)
    )
    await store.save(
        call("experimental.channel.s", now - timedelta(minutes=30), usage, "proactive")
    )
    await store.save(
        call("experimental.jev.decision", now - timedelta(minutes=1), {"body": {}})
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        reply = await client.get(
            "/internal/llm-logs/usage", params={"dayStart": day_start.isoformat()}
        )
    assert reply.status_code == 200
    rows = {(r["call_type"], r["purpose"]): r for r in reply.json()["items"]}
    thought = rows["experimental.thought.generate", ""]
    assert (thought["today_calls"], thought["today_tokens"]) == (2, 240)
    assert (thought["week_calls"], thought["week_tokens"]) == (3, 360)
    assert (thought["hour_calls"], thought["hour_tokens"], thought["recent_calls"]) == (
        1,
        120,
        1,
    )
    assert rows["experimental.channel.s", "proactive"]["today_tokens"] == 120
    jev = rows["experimental.jev.decision", ""]
    assert (jev["today_calls"], jev["today_tokens"]) == (1, 0)


@pytest.mark.parametrize("days", [1, 7, 30])
@pytest.mark.asyncio
async def test_expiry_preserves_recent_images_pagination_and_user_memory(
    postgres_pool, days
):
    from fish_memory.db.fishes import FishStore
    from fish_memory.db.messages import DEFAULT_EMBEDDING_SPACE, MessagesStore

    await run_migrations(postgres_pool)
    # Real PostgreSQL fixture only: these chat and memory sentinels must survive.
    fish_id = await FishStore(
        postgres_pool, embedding_space=DEFAULT_EMBEDDING_SPACE
    ).create(name="retention fixture")
    message = await MessagesStore(postgres_pool).insert_user_message(
        fish_id, "keep this conversation"
    )
    fact_id = uuid4()
    await postgres_pool.execute(
        """INSERT INTO facts
        (id,fish_id,session_id,source_message_id,fact_index,text,mentioned_at,entities)
        VALUES ($1,$2,$3,$4,0,'keep this memory',now(),'[]'::jsonb)""",
        fact_id,
        fish_id,
        message.session_id,
        message.id,
    )
    clock = MutableClock(datetime(2026, 10, 1, 12, tzinfo=UTC))
    retention_days = 365
    store = LlmLogStore(
        postgres_pool, retention_days=lambda: retention_days, clock=clock
    )
    image = "data:image/jpeg;base64,c3ludGhldGljLWltYWdl"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": "keep image bytes and text"},
                ],
            }
        ]
    }
    cutoff = clock.now() - timedelta(days=days)

    def call(at, status="completed"):
        return {
            "id": str(uuid4()),
            "call_type": "chat",
            "model": "fixture",
            "request_url": "https://fixture.invalid/v1/chat/completions",
            "status": status,
            "started_at": at,
            "ended_at": None if status == "running" else at,
            "duration_ms": 1,
            "http_status": 200,
            "error": None,
            "source": "host",
            "request": body,
            "response": None if status == "running" else {"image": image},
        }

    expired = call(cutoff - timedelta(microseconds=1))
    expired_running = call(cutoff - timedelta(seconds=1), "running")
    boundary = call(cutoff)
    recent = call(cutoff + timedelta(microseconds=1))
    running = call(clock.now(), "running")
    for value in (expired, expired_running, boundary, recent, running):
        await store.save(value)
    first = await store.list_calls(limit=2)
    assert [str(row["id"]) for row in first["items"]] == [running["id"], recent["id"]]

    retention_days = days
    await store.process_once()
    assert await store.get(UUID(expired["id"])) is None
    assert await store.get(UUID(expired_running["id"])) is None
    # A queued provider completion and a late desktop POST cannot recreate them.
    await store.save({**expired_running, "status": "completed", "response": body})
    app = FastAPI()
    app.state.llm_log_store = store
    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        incoming = {
            **expired,
            "started_at": expired["started_at"].isoformat(),
            "ended_at": expired["ended_at"].isoformat(),
        }
        assert (
            await client.post("/internal/llm-logs", json=incoming)
        ).status_code == 200
    assert await postgres_pool.fetchval("SELECT count(*) FROM llm_call_logs") == 3
    second = await store.list_calls(limit=2, before=first["next_cursor"])
    assert [str(row["id"]) for row in second["items"]] == [boundary["id"]]
    assert second["next_cursor"] is None
    saved = await store.get(UUID(boundary["id"]))
    assert saved["request"] == body
    assert saved["response"] == {"image": image}
    await store.save({**running, "status": "completed", "response": {"image": image}})
    assert (await store.get(UUID(running["id"])))["response"] == {"image": image}
    assert (
        await postgres_pool.fetchval(
            "SELECT content FROM messages WHERE id=$1", message.id
        )
        == "keep this conversation"
    )
    assert (
        await postgres_pool.fetchval("SELECT text FROM facts WHERE id=$1", fact_id)
        == "keep this memory"
    )
