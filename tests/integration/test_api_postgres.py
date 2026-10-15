import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

from fish_memory.agent.replies import (
    AgentReplyService,
    InputReplyClassification,
    LlmReplyClassifier,
    RunResultReplyAction,
    RunResultReplyClassification,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.api.routes import (
    catalog_router,
    internal_router,
    router,
)
from fish_memory.background.fact_processor import FactProcessor
from fish_memory.background.relation_processor import SnapshotItemRelationProcessor
from fish_memory.background.snapshot_processor import DynamicSnapshotProcessor
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import TaskStore
from fish_memory.db.embedding_spaces import FishEmbeddingSpaceStore
from fish_memory.db.entities import EntityStore
from fish_memory.db.facts import FactStore
from fish_memory.db.fishes import FishStore
from fish_memory.db.home_actions import HomeActionStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.performance import PerformanceStore
from fish_memory.db.recall import RecallStore
from fish_memory.db.relations import SnapshotItemRelationStore
from fish_memory.db.sessions import SessionStore
from fish_memory.db.snapshots import SnapshotStore
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.entities.resolver import EntityResolver
from fish_memory.facts.extractor import FactExtractionResult
from fish_memory.facts.models import ExtractedFact
from fish_memory.llm.client import ChatStreamChunk, JsonCompletion
from fish_memory.models import Message, MessageStatus
from fish_memory.performance.models import JsonValue
from fish_memory.recall.service import FactRecallService
from fish_memory.snapshots.confirmer import SnapshotUpdateConfirmer
from fish_memory.snapshots.updater import DynamicSnapshotUpdater


class FakeChatClient:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages, enable_search, force_search
        for chunk in self.chunks:
            yield ChatStreamChunk(content=chunk)

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages, enable_search, force_search
        yield ChatStreamChunk(content='{"m":"neutral","a":"')
        for chunk in self.chunks:
            yield ChatStreamChunk(content=chunk)
        yield ChatStreamChunk(content='","r":0,"e":0}')

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search, force_search
        return {
            "model": "fake",
            "messages": [dict(message) for message in messages],
            "parameters": {"stream": True},
        }

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search, force_search
        return self.request_payload(messages)

    async def close(self) -> None:
        return None


class ForbiddenReplyJsonClient:
    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        del messages
        raise AssertionError("explicit approval must not call the reply classifier")

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": [dict(message) for message in messages]}


class RecordingReplyCoordinator:
    def __init__(self) -> None:
        self.user_notifications = 0
        self.voice_notifications = 0

    def notify_user_message(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.user_notifications += 1

    def notify_voice_turn(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.voice_notifications += 1


class ScriptedReplyClassifier:
    def __init__(self, *actions: RunResultReplyAction) -> None:
        self.actions = list(actions)
        self.calls = 0

    async def classify_input(
        self, *, request: str, content: str
    ) -> InputReplyClassification:
        del request, content
        raise AssertionError("result test must not classify input")

    async def classify_run_result(
        self, *, result: str, content: str
    ) -> RunResultReplyClassification:
        del result, content
        self.calls += 1
        return RunResultReplyClassification(self.actions.pop(0))


class BlockingDivergentReplyClassifier(ScriptedReplyClassifier):
    def __init__(self) -> None:
        super().__init__(
            RunResultReplyAction.CONTINUE,
            RunResultReplyAction.UNRELATED,
        )
        self.two_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def classify_run_result(
        self, *, result: str, content: str
    ) -> RunResultReplyClassification:
        del result, content
        index = self.calls
        self.calls += 1
        if self.calls == 2:
            self.two_entered.set()
        await self.release.wait()
        return RunResultReplyClassification(self.actions[index])


class RecallLaneChatClient(FakeChatClient):
    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages, enable_search, force_search
        yield ChatStreamChunk(
            content=(
                '{"m":"neutral","a":"让我想想。","r":1,"q":"用户的近期偏好","e":0}'
            )
        )


class FailingRecallPlanner:
    async def plan(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("planner unavailable")


class RecallHeadSpy:
    def __init__(self, store: RecallStore) -> None:
        self._store = store
        self.fact_head_calls = 0
        self.relation_head_calls = 0

    async def search_fact_head(self, *args: Any, **kwargs: Any) -> object:
        self.fact_head_calls += 1
        return await self._store.search_fact_head(*args, **kwargs)

    async def search_relation_head(self, *args: Any, **kwargs: Any) -> object:
        self.relation_head_calls += 1
        return await self._store.search_relation_head(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)


class FakeFactExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str]] = []

    async def extract(
        self,
        message: Message,
        content: str,
        context: object | None = None,
    ) -> FactExtractionResult:
        self.calls.append((message.id, content))
        return FactExtractionResult(
            facts=(
                ExtractedFact(
                    text="主人住在上海",
                    occurred_start=None,
                    occurred_end=None,
                    entities=("主人", "上海"),
                ),
            ),
            request_payload={"model": "fake-fact-extractor"},
            response_payload={"content": "主人住在上海"},
        )


class FakeEmbedder:
    space = EmbeddingSpace(
        provider="local",
        model="intfloat/multilingual-e5-small",
        dimensions=1024,
    )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=tuple((0.25,) * 1024 for _ in texts),
            input_tokens=len(texts) * 3,
        )

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult:
        return await self.embed(texts)


class RecallEmbedder:
    space = EmbeddingSpace(
        provider="local",
        model="intfloat/multilingual-e5-small",
        dimensions=384,
    )

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=tuple((0.25,) * 384 for _ in texts),
            input_tokens=len(texts) * 3,
        )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return await self.embed_query(texts)


class ScriptedSnapshotChatClient:
    def __init__(
        self,
        *,
        preupdates: Sequence[Mapping[str, object]],
        confirmations: Sequence[Mapping[str, object]],
    ) -> None:
        self._preupdates = [dict(item) for item in preupdates]
        self._confirmations = [dict(item) for item in confirmations]
        self.preupdate_payloads: list[dict[str, object]] = []
        self.confirmation_payloads: list[dict[str, object]] = []

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        system = messages[0]["content"]
        payload = cast(dict[str, object], json.loads(messages[1]["content"]))
        if system.startswith("你负责为 Dynamic 短期记忆快照执行第一阶段预更新"):
            self.preupdate_payloads.append(payload)
            return JsonCompletion(
                json.dumps(
                    self._preupdates.pop(0),
                    ensure_ascii=False,
                )
            )
        if system.startswith("你负责确认一次已有快照条目的预更新标记"):
            self.confirmation_payloads.append(payload)
            return JsonCompletion(
                json.dumps(
                    self._confirmations.pop(0),
                    ensure_ascii=False,
                )
            )
        if system.startswith("你负责判断不同候选是否指向同一个真实对象"):
            return JsonCompletion('{"groups":[]}')
        raise AssertionError(f"unexpected LLM prompt: {system[:80]}")

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": [dict(message) for message in messages]}


class SnapshotEvolutionEmbedder:
    space = EmbeddingSpace("local", "snapshot-evolution-test", 3)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0, 0.0) for _ in texts),
            input_tokens=len(texts),
        )


def snapshot_processors(
    pool: asyncpg.Pool,
    llm: ScriptedSnapshotChatClient,
    embedder: SnapshotEvolutionEmbedder,
) -> tuple[DynamicSnapshotProcessor, SnapshotItemRelationProcessor]:
    snapshot_store = SnapshotStore(pool)
    return (
        DynamicSnapshotProcessor(
            snapshot_store,
            DynamicSnapshotUpdater(llm),
            SnapshotUpdateConfirmer(llm),
            EntityResolver(llm),
            EntityStore(pool),
            PerformanceStore(pool),
            llm_semaphore=asyncio.Semaphore(2),
            max_retries=1,
            batch_size=20,
            now=lambda: datetime(2026, 8, 27, 12, tzinfo=UTC),
        ),
        SnapshotItemRelationProcessor(
            SnapshotItemRelationStore(pool, snapshot_type="dynamic"),
            embedder,
            embedding_space_store=FishEmbeddingSpaceStore(pool),
            embedding_semaphore=asyncio.Semaphore(1),
            snapshot_type="dynamic",
            max_retries=1,
            batch_size=20,
        ),
    )


async def persist_pending_facts(
    pool: asyncpg.Pool,
    facts: Sequence[ExtractedFact],
) -> tuple[UUID, ...]:
    store = FactStore(pool)
    pending = (await store.list_pending_messages(limit=10))[0]
    await store.persist_extraction(
        pending,
        facts,
        tuple((1.0, 0.0, 0.0) for _ in facts),
        SnapshotEvolutionEmbedder.space,
    )
    rows = await pool.fetch(
        "SELECT id FROM facts WHERE source_message_id = $1 ORDER BY fact_index",
        pending.message.id,
    )
    return tuple(row["id"] for row in rows)


@pytest.mark.asyncio
async def test_ask_generation_sse_and_progress_with_postgres(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    performance_store = PerformanceStore(postgres_pool)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["你好", "🐟"]),
        performance_store=performance_store,
        system_prompt="system",
    )
    app = FastAPI()
    app.state.message_store = store
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.state.performance_store = performance_store
    app.state.home_action_store = HomeActionStore(postgres_pool)
    app.include_router(catalog_router)
    app.include_router(router)
    app.include_router(internal_router)
    fish_id = uuid4()
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            ask_response = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={"content": "在吗？"},
            )
            assert ask_response.status_code == 201
            user_message_id = UUID(ask_response.json()["messageId"])
            session_id = UUID(ask_response.json()["sessionId"])

            runtime = coordinator.get_runtime(fish_id, session_id)
            async with runtime.condition:
                await asyncio.wait_for(
                    runtime.condition.wait_for(lambda: runtime.llm_task is not None),
                    timeout=1,
                )
            assert runtime.llm_task is not None
            await runtime.llm_task
            assert runtime.llm_task.exception() is None
            assert runtime.status is MessageStatus.COMPLETED

            answer_response = await client.get(f"/v1/fishes/{fish_id}/answer")
            assert answer_response.status_code == 200
            assert "answer.completed" in answer_response.text
            assert "你好🐟" in answer_response.text

            assert runtime.answer_message_id is not None
            progress_response = await client.post(
                f"/v1/fishes/{fish_id}/answer/progress",
                json={
                    "messageId": str(runtime.answer_message_id),
                    "consumedOffset": 1,
                },
            )
            assert progress_response.status_code == 204

        async with postgres_pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT role, content, status, consumed_offset
                FROM messages
                WHERE fish_id = $1
                ORDER BY sequence
                """,
                fish_id,
            )
            snapshot_storage_version = await connection.fetchval(
                "SELECT snapshot_storage_version FROM fishes WHERE id = $1",
                fish_id,
            )
        assert [(row["role"], row["status"]) for row in rows] == [
            ("user", "initialized"),
            ("assistant", "completed"),
        ]
        assert rows[0]["content"] == "在吗？"
        assert rows[1]["consumed_offset"] == 1
        assert snapshot_storage_version == 2

        performance_steps = []
        for _ in range(100):
            performance_steps = await performance_store.list_steps(user_message_id)
            if performance_steps:
                break
            await asyncio.sleep(0.01)

        by_name = {step.name: step for step in performance_steps}
        assert {
            "answer.first_content",
            "llm.first_content",
            "llm.answer.initial",
            "total",
        } <= set(by_name)
        assert all(step.duration_us >= 0 for step in performance_steps)
        initial_payload = cast(
            dict[str, object], by_name["llm.answer.initial"].input_payload
        )
        initial_messages = cast(list[dict[str, str]], initial_payload["messages"])
        assert initial_payload["model"] == "fake"
        assert initial_payload["parameters"] == {"stream": True}
        assert initial_messages[0]["role"] == "system"
        assert initial_messages[0]["content"].startswith("# R 通道输出协议")
        assert initial_messages[-1]["role"] == "user"
        assert '"content":"<message_time>' in initial_messages[-1]["content"]
        assert "\\n在吗？" in initial_messages[-1]["content"]
        assert "# 当前问题的 R 通道复核" in initial_messages[-1]["content"]
        assert by_name["llm.answer.initial"].output_payload == {
            "content": '{"m":"neutral","a":"你好🐟","r":0,"e":0}',
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
        }

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            performance_response = await client.get(
                f"/internal/fishes/{fish_id}/messages/{user_message_id}/performance"
            )
            cross_fish_response = await client.get(
                f"/internal/fishes/{uuid4()}/messages/{user_message_id}/performance"
            )
            delete_response = await client.delete(f"/v1/fishes/{fish_id}")
            list_response = await client.get("/v1/fishes")
            history_response = await client.get(f"/v1/fishes/{fish_id}/messages")
            ask_deleted_response = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={"content": "回来"},
            )

        assert performance_response.status_code == 200
        llm_step = next(
            item
            for item in performance_response.json()["items"]
            if item["stepName"] == "llm.answer.initial"
        )
        assert llm_step["inputPayload"]["messages"] == initial_messages
        assert (
            llm_step["outputPayload"]["content"]
            == '{"m":"neutral","a":"你好🐟","r":0,"e":0}'
        )
        assert cross_fish_response.status_code == 200
        assert cross_fish_response.json() == {"items": [], "homeActions": []}
        assert delete_response.status_code == 204
        assert list_response.json()["items"] == []
        assert history_response.json()["items"] == []
        assert ask_deleted_response.status_code == 410

        async with postgres_pool.acquire() as connection:
            message_count = await connection.fetchval(
                "SELECT count(*) FROM messages WHERE fish_id = $1",
                fish_id,
            )
            performance_count = await connection.fetchval(
                """
                SELECT count(*)
                FROM message_performance_steps
                WHERE message_id = $1
                """,
                user_message_id,
            )
        assert message_count == 2
        assert performance_count > 0
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_exact_approval_api_is_atomic_and_skips_ordinary_answer(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "start task")
    target = await targets.upsert_target(
        "hermes",
        "http://localhost:9000",
        "default",
        "",
        "credential-ref",
        None,
        "Hermes",
    )
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "task",
        source.id,
        None,
        source.content,
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    assert await tasks.record_dispatch_accepted(run.id, "external-run", {}) is not None
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["once", "deny"]}
    )
    request_message = await tasks.materialize_interaction_request(
        interaction.id, "Approve this operation?"
    )
    assert request_message is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request_message.id,
            fish_id,
            request_message.session_id,
            len(request_message.content),
            False,
        )
        is not None
    )
    coordinator = AnswerCoordinator(
        messages,
        FakeChatClient(["must not answer"]),
        system_prompt="system",
    )
    app = FastAPI()
    app.state.message_store = messages
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.state.agent_reply_service = AgentReplyService(
        tasks,
        LlmReplyClassifier(ForbiddenReplyJsonClient()),
    )
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={
                    "content": "允许",
                    "sessionId": str(source.session_id),
                    "replyContext": {
                        "kind": "interaction",
                        "id": str(interaction.id),
                    },
                },
            )
            duplicate = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={
                    "content": "允许",
                    "sessionId": str(source.session_id),
                    "replyContext": {
                        "kind": "interaction",
                        "id": str(interaction.id),
                    },
                },
            )

        assert response.status_code == 201
        assert response.json()["answerExpected"] is False
        assert duplicate.status_code == 409
        stored = await postgres_pool.fetchrow(
            """
            SELECT interaction.response_message_id,
                   interaction.provider_data #>>
                       '{approval_response,choice}' AS choice,
                   interaction.provider_data #>>
                       '{approval_response,resolve_all}' AS resolve_all
            FROM agent_interactions AS interaction
            WHERE interaction.id = $1
            """,
            interaction.id,
        )
        assert stored is not None
        assert stored["response_message_id"] == UUID(response.json()["messageId"])
        assert stored["choice"] == "once"
        assert stored["resolve_all"] == "false"
        assert (
            await postgres_pool.fetchval(
                """
                SELECT count(*)
                FROM messages
                WHERE fish_id = $1 AND role = 'user' AND content = '允许'
                """,
                fish_id,
            )
            == 1
        )
        assert coordinator.peek_runtime(fish_id, source.session_id) is None
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_exact_voice_replay_and_concurrent_ask_have_one_durable_winner(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    fish_id = uuid4()
    first_source = await messages.insert_user_message(fish_id, "first task")
    target = await targets.upsert_target(
        "hermes",
        "http://localhost:9000",
        "default",
        "",
        "credential-ref",
        None,
        "Hermes",
    )

    async def approval_subject(label: str) -> UUID:
        source = await messages.insert_user_message(
            fish_id,
            label,
            session_id=first_source.session_id,
        )
        _task, run = await tasks.create_task_with_first_run(
            fish_id,
            target.target.id,
            label,
            source.id,
            None,
            source.content,
        )
        assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
        assert (
            await tasks.record_dispatch_accepted(run.id, f"external-{label}", {})
            is not None
        )
        interaction = await tasks.open_approval_interaction(
            run.id, {"allowed_decisions": ["once", "deny"]}
        )
        request_message = await tasks.materialize_interaction_request(
            interaction.id, f"Approve {label}?"
        )
        assert request_message is not None
        assert (
            await tasks.finalize_agent_message_consumption(
                request_message.id,
                fish_id,
                request_message.session_id,
                len(request_message.content),
                False,
            )
            is not None
        )
        return interaction.id

    voice_interaction_id = await approval_subject("voice")
    concurrent_interaction_id = await approval_subject("concurrent")
    coordinator = AnswerCoordinator(
        messages,
        FakeChatClient(["must not answer"]),
        system_prompt="system",
    )
    app = FastAPI()
    app.state.message_store = messages
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.state.agent_reply_service = AgentReplyService(
        tasks,
        LlmReplyClassifier(ForbiddenReplyJsonClient()),
    )
    app.include_router(router)
    app.include_router(internal_router)
    transport = httpx.ASGITransport(app=app)
    turn_id = uuid4()
    voice_path = (
        f"/internal/fishes/{fish_id}/sessions/{first_source.session_id}"
        f"/voice-turns/{turn_id}/commit"
    )
    voice_body = {
        "revision": 1,
        "content": "允许",
        "replyContext": {
            "kind": "interaction",
            "id": str(voice_interaction_id),
        },
    }
    ask_body = {
        "content": "允许",
        "sessionId": str(first_source.session_id),
        "replyContext": {
            "kind": "interaction",
            "id": str(concurrent_interaction_id),
        },
    }

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            first_voice = await client.post(voice_path, json=voice_body)
            replay_voice = await client.post(voice_path, json=voice_body)
            concurrent = await asyncio.gather(
                client.post(f"/v1/fishes/{fish_id}/ask", json=ask_body),
                client.post(f"/v1/fishes/{fish_id}/ask", json=ask_body),
            )

        assert first_voice.status_code == replay_voice.status_code == 201
        assert first_voice.json() == replay_voice.json()
        assert first_voice.json()["messageId"] == str(turn_id)
        assert first_voice.json()["answerExpected"] is False
        assert sorted(response.status_code for response in concurrent) == [201, 409]
        assert (
            next(
                response for response in concurrent if response.status_code == 201
            ).json()["answerExpected"]
            is False
        )
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM messages WHERE id = $1", turn_id
            )
            == 1
        )
        assert (
            await postgres_pool.fetchval(
                """
                SELECT count(*)
                FROM messages
                WHERE fish_id = $1 AND role = 'user' AND content = '允许'
                """,
                fish_id,
            )
            == 2
        )
        assert coordinator.peek_runtime(fish_id, first_source.session_id) is None
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_t_confirmed_child_keeps_ordinary_voice_replay_answer_expected(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "start")
    target = await targets.upsert_target(
        "hermes",
        "http://localhost:9000",
        "default",
        "",
        "credential-ref",
        None,
        "Hermes",
    )
    task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "task",
        source.id,
        None,
        source.content,
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    assert await tasks.record_dispatch_accepted(run.id, "external-run", {}) is not None
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    classifier = ScriptedReplyClassifier(RunResultReplyAction.UNRELATED)
    coordinator = RecordingReplyCoordinator()
    app = FastAPI()
    app.state.message_store = messages
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.state.agent_reply_service = AgentReplyService(tasks, classifier)
    app.include_router(internal_router)
    turn_id = uuid4()
    path = (
        f"/internal/fishes/{fish_id}/sessions/{source.session_id}"
        f"/voice-turns/{turn_id}/commit"
    )
    body = {
        "revision": 1,
        "content": "continue",
        "replyContext": {"kind": "run_result", "id": str(run.id)},
    }
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(path, json=body)
        confirmation = await messages.insert_generating_answer(
            fish_id,
            source.session_id,
            int(first.json()["sequence"]),
        )
        child = await tasks.create_continuation_run(
            task.id,
            run.id,
            turn_id,
            "continue",
            confirmation.id,
        )
        replay = await client.post(path, json=body)

    assert child.confirmation_message_id == confirmation.id
    assert first.status_code == replay.status_code == 201
    assert first.json() == replay.json()
    assert replay.json()["answerExpected"] is True
    assert classifier.calls == 1
    assert coordinator.voice_notifications == 1


@pytest.mark.asyncio
async def test_concurrent_divergent_classifiers_share_final_store_disposition(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "start")
    target = await targets.upsert_target(
        "hermes",
        "http://localhost:9000",
        "default",
        "",
        "credential-ref",
        None,
        "Hermes",
    )
    task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "task",
        source.id,
        None,
        source.content,
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    assert await tasks.record_dispatch_accepted(run.id, "external-run", {}) is not None
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    classifier = BlockingDivergentReplyClassifier()
    coordinator = RecordingReplyCoordinator()
    app = FastAPI()
    app.state.message_store = messages
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.state.agent_reply_service = AgentReplyService(tasks, classifier)
    app.include_router(internal_router)
    turn_id = uuid4()
    path = (
        f"/internal/fishes/{fish_id}/sessions/{source.session_id}"
        f"/voice-turns/{turn_id}/commit"
    )
    body = {
        "revision": 1,
        "content": "continue",
        "replyContext": {"kind": "run_result", "id": str(run.id)},
    }
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first_request = asyncio.create_task(client.post(path, json=body))
        second_request = asyncio.create_task(client.post(path, json=body))
        await asyncio.wait_for(classifier.two_entered.wait(), timeout=1)
        classifier.release.set()
        first, second = await asyncio.gather(first_request, second_request)

    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    answer_expected = bool(first.json()["answerExpected"])
    assert coordinator.voice_notifications == (1 if answer_expected else 0)
    assert classifier.calls == 2
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1", turn_id
        )
        == 1
    )
    assert await postgres_pool.fetchval(
        """
            SELECT count(*)
            FROM agent_runs
            WHERE task_id = $1 AND previous_run_id = $2
            """,
        task.id,
        run.id,
    ) == (0 if answer_expected else 1)


@pytest.mark.asyncio
async def test_recall_generation_searches_each_head_once_with_postgres(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    recall_store = RecallHeadSpy(RecallStore(postgres_pool))
    recall_service = FactRecallService(
        recall_store,  # ty: ignore[invalid-argument-type]
        FailingRecallPlanner(),  # ty: ignore[invalid-argument-type]
        RecallEmbedder(),
        FishEmbeddingSpaceStore(postgres_pool),
    )
    coordinator = AnswerCoordinator(
        store,
        RecallLaneChatClient([]),
        recall_service=recall_service,
        system_prompt="system",
    )
    app = FastAPI()
    app.state.message_store = store
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.include_router(router)
    fish_id = uuid4()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={"content": "我最近喜欢什么？"},
            )
        assert response.status_code == 201
        session_id = UUID(response.json()["sessionId"])
        runtime = coordinator.get_runtime(fish_id, session_id)
        async with runtime.condition:
            await asyncio.wait_for(
                runtime.condition.wait_for(lambda: runtime.llm_task is not None),
                timeout=1,
            )
        assert runtime.llm_task is not None
        await runtime.llm_task

        assert recall_store.fact_head_calls == 1
        assert recall_store.relation_head_calls == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_ask_rejects_legacy_fish_without_creating_chat_rows(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["unused"]),
        system_prompt="system",
    )
    fish_id = uuid4()
    await postgres_pool.execute(
        """
        INSERT INTO fishes (id, snapshot_storage_version)
        VALUES ($1, 1)
        """,
        fish_id,
    )
    app = FastAPI()
    app.state.message_store = store
    app.state.session_store = SessionStore(postgres_pool)
    app.state.answer_coordinator = coordinator
    app.include_router(router)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/v1/fishes/{fish_id}/ask",
                json={"content": "legacy"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "legacy_fish_read_only"
        counts = await postgres_pool.fetchrow(
            """
            SELECT
                (SELECT count(*) FROM fish_sessions WHERE fish_id = $1) AS sessions,
                (SELECT count(*) FROM messages WHERE fish_id = $1) AS messages
            """,
            fish_id,
        )
        assert counts is not None
        assert counts["sessions"] == 0
        assert counts["messages"] == 0
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_fact_processing_progress_with_postgres(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    message_store = MessagesStore(
        postgres_pool,
        embedding_space=FakeEmbedder.space,
    )
    fact_store = FactStore(postgres_pool)
    performance_store = PerformanceStore(postgres_pool)
    extractor = FakeFactExtractor()
    processor = FactProcessor(
        fact_store,
        extractor,
        FakeEmbedder(),
        performance_store,
        embedding_space_store=FishEmbeddingSpaceStore(postgres_pool),
        llm_semaphore=asyncio.Semaphore(1),
        embedding_semaphore=asyncio.Semaphore(1),
        max_retries=3,
        batch_size=8,
    )
    app = FastAPI()
    app.state.fact_store = fact_store
    app.state.session_store = SessionStore(postgres_pool)
    app.include_router(internal_router)
    fish_id = uuid4()
    first = await message_store.insert_user_message(fish_id, "我住在上海")
    latest = await message_store.insert_user_message(fish_id, "这是最新消息")

    candidates = await fact_store.list_pending_messages(limit=10)

    assert [candidate.message.id for candidate in candidates] == [first.id]
    assert candidates[0].target_sequence == first.sequence
    assert latest.sequence > candidates[0].target_sequence
    assert await processor.process_once() is True
    assert extractor.calls == [(first.id, "我住在上海")]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(f"/internal/fishes/{fish_id}/background-progress")

    assert response.status_code == 200
    assert response.json()["tasks"] == [
        {
            "name": "fact_extraction",
            "throughSequence": first.sequence,
            "targetSequence": first.sequence,
            "pending": 0,
            "lastError": None,
            "updatedAt": response.json()["tasks"][0]["updatedAt"],
        }
    ]
    assert response.json()["tasks"][0]["updatedAt"] is not None

    fact = await postgres_pool.fetchrow(
        """
        SELECT
            fact.source_message_id,
            fact.fact_index,
            fact.text,
            ARRAY(
                SELECT jsonb_array_elements_text(fact.entities)
            ) AS entities
        FROM facts AS fact
        WHERE fact.fish_id = $1
        """,
        fish_id,
    )
    assert fact is not None
    assert fact["source_message_id"] == first.id
    assert fact["fact_index"] == 0
    assert fact["text"] == "主人住在上海"
    assert fact["entities"] == ["主人", "上海"]


@pytest.mark.asyncio
async def test_wrong_snapshot_update_confirmation_creates_an_independent_item(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches a rejected update overwriting an unrelated current Item or Full."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=SnapshotEvolutionEmbedder.space,
    ).create(name="snapshot-confirm-create")
    messages = MessagesStore(
        postgres_pool,
        embedding_space=SnapshotEvolutionEmbedder.space,
    )
    first_source = await messages.insert_user_message(
        fish_id,
        "我的衣柜还有几项商店事项",
        created_at=datetime(2026, 8, 27, 10, tzinfo=UTC),
    )
    await messages.insert_user_message(
        fish_id,
        "我开始制作书籍记忆墙",
        session_id=first_source.session_id,
        created_at=datetime(2026, 8, 27, 10, 1, tzinfo=UTC),
    )
    wardrobe_fact_ids = await persist_pending_facts(
        postgres_pool,
        (
            ExtractedFact(
                text="用户需要从商店取回西装和新靴。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
            ExtractedFact(
                text="用户需要把旧靴退回商店。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
        ),
    )
    llm = ScriptedSnapshotChatClient(
        preupdates=(
            {
                "creates": [
                    {
                        "summary": "衣柜事项：西装和新靴待取，旧靴待退。",
                        "sourceFactIndexes": [0, 1],
                    }
                ],
                "updates": [],
            },
            {
                "creates": [],
                "updates": [{"itemIndex": 0, "sourceFactIndexes": [0]}],
            },
        ),
        confirmations=(
            {
                "operations": [
                    {
                        "action": "create",
                        "basis": "书籍记忆墙是与衣柜事项无关的新主题。",
                        "newSupportIndexes": [0],
                        "summary": "用户开始制作书籍记忆墙。",
                    }
                ]
            },
        ),
    )
    embedder = SnapshotEvolutionEmbedder()
    snapshot_processor, relation_processor = snapshot_processors(
        postgres_pool,
        llm,
        embedder,
    )

    assert await snapshot_processor.process_once() is True
    assert await relation_processor.process_once() is True
    assert embedder.calls == [("衣柜事项：西装和新靴待取，旧靴待退。",)]
    snapshot_store = SnapshotStore(postgres_pool)
    initial = await snapshot_store.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    assert initial is not None
    assert len(initial.items) == 1
    wardrobe = initial.items[0]
    assert wardrobe.support_fact_ids == wardrobe_fact_ids
    wardrobe_full_before = await postgres_pool.fetchrow(
        """
        SELECT item_version, snapshot_text, text_hash, embedding::text AS vector
        FROM memory_snapshot_item_embeddings
        WHERE snapshot_item_id = $1
        """,
        wardrobe.id,
    )
    assert wardrobe_full_before is not None

    await messages.insert_user_message(
        fish_id,
        "保持前一条消息稳定",
        session_id=first_source.session_id,
        created_at=datetime(2026, 8, 27, 10, 2, tzinfo=UTC),
    )
    (book_fact_id,) = await persist_pending_facts(
        postgres_pool,
        (
            ExtractedFact(
                text="用户开始制作书籍记忆墙。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
        ),
    )

    assert await snapshot_processor.process_once() is True
    assert await relation_processor.process_once() is True
    assert embedder.calls == [
        ("衣柜事项：西装和新靴待取，旧靴待退。",),
        ("用户开始制作书籍记忆墙。",),
    ]

    current = await snapshot_store.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    assert current is not None
    current_by_summary = {item.summary: item for item in current.items}
    current_wardrobe = current_by_summary[wardrobe.summary]
    book = current_by_summary["用户开始制作书籍记忆墙。"]
    assert current.version == 2
    assert current_wardrobe == wardrobe
    assert book.support_fact_ids == (book_fact_id,)
    assert book.lineage_fact_ids == (book_fact_id,)
    wardrobe_full_after = await postgres_pool.fetchrow(
        """
        SELECT item_version, snapshot_text, text_hash, embedding::text AS vector
        FROM memory_snapshot_item_embeddings
        WHERE snapshot_item_id = $1
        """,
        wardrobe.id,
    )
    assert dict(wardrobe_full_after or {}) == dict(wardrobe_full_before)
    full_rows = await postgres_pool.fetch(
        """
        SELECT projection.snapshot_item_id, projection.item_version,
               projection.snapshot_text, item.support_count
        FROM memory_snapshot_item_embeddings AS projection
        JOIN memory_snapshot_items AS item ON item.id = projection.snapshot_item_id
        WHERE projection.fish_id = $1
        ORDER BY projection.snapshot_item_id
        """,
        fish_id,
    )
    assert [
        (
            row["snapshot_item_id"],
            row["item_version"],
            row["snapshot_text"],
            row["support_count"],
        )
        for row in full_rows
    ] == [
        (wardrobe.id, 1, wardrobe.summary, 2),
        (book.id, 1, book.summary, 1),
    ]
    assert llm.confirmation_payloads[0]["targetItem"] == {
        "itemIndex": 0,
        "summary": wardrobe.summary,
    }
    assert len(cast(list[object], llm.confirmation_payloads[0]["supportFacts"])) == 2


@pytest.mark.asyncio
async def test_partial_topic_confirmation_updates_and_creates_in_one_revision(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches split confirmation becoming two Header versions or losing lineage."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = await FishStore(
        postgres_pool,
        embedding_space=SnapshotEvolutionEmbedder.space,
    ).create(name="snapshot-confirm-split")
    messages = MessagesStore(
        postgres_pool,
        embedding_space=SnapshotEvolutionEmbedder.space,
    )
    first_source = await messages.insert_user_message(
        fish_id,
        "我有两个相关的初始进展",
        created_at=datetime(2026, 8, 27, 11, tzinfo=UTC),
    )
    await messages.insert_user_message(
        fish_id,
        "一个进展更新，另一个是独立事项",
        session_id=first_source.session_id,
        created_at=datetime(2026, 8, 27, 11, 1, tzinfo=UTC),
    )
    old_fact_ids = await persist_pending_facts(
        postgres_pool,
        (
            ExtractedFact(
                text="旧主题的临时计划。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
            ExtractedFact(
                text="旧主题中仍然有效的约束。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
        ),
    )
    llm = ScriptedSnapshotChatClient(
        preupdates=(
            {
                "creates": [
                    {
                        "summary": "旧主题包含临时计划和持续约束。",
                        "sourceFactIndexes": [0, 1],
                    }
                ],
                "updates": [],
            },
            {
                "creates": [],
                "updates": [{"itemIndex": 0, "sourceFactIndexes": [0, 1]}],
            },
        ),
        confirmations=(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "新状态替换临时计划，并保留持续约束。",
                        "itemIndex": 0,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [0],
                        "summary": "旧主题保留持续约束，并采用新的当前状态。",
                    },
                    {
                        "action": "create",
                        "basis": "第二条新事实属于独立主题。",
                        "newSupportIndexes": [1],
                        "summary": "用户开始了一个独立的新主题。",
                    },
                ]
            },
        ),
    )
    embedder = SnapshotEvolutionEmbedder()
    snapshot_processor, relation_processor = snapshot_processors(
        postgres_pool,
        llm,
        embedder,
    )

    assert await snapshot_processor.process_once() is True
    assert await relation_processor.process_once() is True
    assert embedder.calls == [("旧主题包含临时计划和持续约束。",)]
    snapshot_store = SnapshotStore(postgres_pool)
    before = await snapshot_store.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    assert before is not None
    old_item = before.items[0]

    await messages.insert_user_message(
        fish_id,
        "保持前一条消息稳定",
        session_id=first_source.session_id,
        created_at=datetime(2026, 8, 27, 11, 2, tzinfo=UTC),
    )
    new_fact_ids = await persist_pending_facts(
        postgres_pool,
        (
            ExtractedFact(
                text="旧主题采用新的当前状态。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
            ExtractedFact(
                text="用户开始了一个独立的新主题。",
                occurred_start=None,
                occurred_end=None,
                entities=(),
            ),
        ),
    )

    assert await snapshot_processor.process_once() is True
    assert await relation_processor.process_once() is True
    assert embedder.calls == [
        ("旧主题包含临时计划和持续约束。",),
        (
            "旧主题保留持续约束，并采用新的当前状态。",
            "用户开始了一个独立的新主题。",
        ),
    ]

    current = await snapshot_store.load_normalized_dynamic_snapshot(
        fish_id,
        first_source.session_id,
    )
    assert current is not None
    assert current.version == before.version + 1
    assert len(current.items) == 2
    updated = next(item for item in current.items if item.id == old_item.id)
    created = next(item for item in current.items if item.id != old_item.id)
    assert updated.summary == "旧主题保留持续约束，并采用新的当前状态。"
    assert updated.support_fact_ids == (old_fact_ids[1], new_fact_ids[0])
    assert updated.lineage_fact_ids == (*old_fact_ids, new_fact_ids[0])
    assert updated.item_version == 2
    assert created.summary == "用户开始了一个独立的新主题。"
    assert created.support_fact_ids == (new_fact_ids[1],)
    assert created.lineage_fact_ids == (new_fact_ids[1],)
    assert created.item_version == 1
    header = await postgres_pool.fetchrow(
        """
        SELECT id, version
        FROM memory_snapshots
        WHERE fish_id = $1 AND session_id = $2 AND snapshot_type = 'dynamic'
        """,
        fish_id,
        first_source.session_id,
    )
    assert header is not None
    assert header["version"] == 2
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_revisions WHERE snapshot_id = $1",
            header["id"],
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
            header["id"],
        )
        == 3
    )
    projections = await postgres_pool.fetch(
        """
        SELECT projection.snapshot_item_id, projection.item_version,
               projection.snapshot_text, item.support_count
        FROM memory_snapshot_item_embeddings AS projection
        JOIN memory_snapshot_items AS item ON item.id = projection.snapshot_item_id
        WHERE projection.fish_id = $1
        ORDER BY projection.snapshot_item_id
        """,
        fish_id,
    )
    assert {
        (
            row["snapshot_item_id"],
            row["item_version"],
            row["snapshot_text"],
            row["support_count"],
        )
        for row in projections
    } == {
        (updated.id, 2, updated.summary, 2),
        (created.id, 1, created.summary, 1),
    }
