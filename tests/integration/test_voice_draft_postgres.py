from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import asyncpg
import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from fish_memory.agent.routing import (
    AgentTaskRouteService,
    LocalConnectorCapabilityPolicy,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.parallel_generation import ParallelAnswerGenerationPipeline
from fish_memory.answer.sse import iter_answer_events
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.answer.voice_drafts import (
    VoiceDraftCoordinator,
)
from fish_memory.api.routes import catalog_router, internal_router, router
from fish_memory.app import _voice_draft_preparer
from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.companion.client import ApiCompanionClient
from fish_memory.companion.draft_tts import (
    DraftTtsSession,
    PromotablePcmSink,
)
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.tts import TtsConfiguration, TtsProviderState
from fish_memory.companion.voice_drafts import VoiceDraftTurnController
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import TaskStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.sessions import SessionStore
from fish_memory.home.models import (
    HomeActionStatus,
    HomeIntentType,
    HomeRunOutcome,
)
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import visible_message_content
from fish_memory.performance.models import JsonValue

FISH_ID = UUID("00000000-0000-0000-0000-000000000411")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000412")
ROLLBACK_TURN_ID = UUID("00000000-0000-0000-0000-000000000413")
TURN_ID = UUID("00000000-0000-0000-0000-000000000414")
TASK_TURN_ID = UUID("00000000-0000-0000-0000-000000000415")
CONTENT = "打开客厅灯"
TASK_CONTENT = "调查构建失败并整理修复建议"
BASE_PATH = f"/internal/fishes/{FISH_ID}/sessions/{SESSION_ID}/voice-turns"
COMMIT_PATH = f"{BASE_PATH}/{TURN_ID}/commit"
COMMIT_BODY = {
    "revision": 2,
    "promoteRevision": 1,
    "content": CONTENT,
}


class RecordingChatClient:
    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        system = messages[0]["content"]
        is_task = TASK_CONTENT in messages[-1]["content"]
        if system.startswith("# H 通道"):
            content = (
                '{"h":0,"m":"neutral","a":""}'
                if is_task
                else '{"h":1,"t":"control","m":"happy","a":"已经打开。"}'
            )
            yield ChatStreamChunk(content=content)
        elif system.startswith("# W 通道"):
            yield ChatStreamChunk(content='{"w":0,"m":"neutral","a":"稍等。"}')
        elif system.startswith("# T 通道"):
            content = (
                '{"t":1,"m":"happy","a":"我来调查并整理修复建议。"}'
                if is_task
                else '{"t":0,"m":"neutral","a":""}'
            )
            yield ChatStreamChunk(content=content)
        else:
            yield ChatStreamChunk(content='{"m":"neutral","a":"稍等。","r":0,"e":0}')

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        async for chunk in self.stream_json(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        ):
            yield chunk

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search, force_search
        return {
            "model": "voice-draft-postgres-fake",
            "messages": [dict(message) for message in messages],
        }

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        return self.json_stream_request_payload(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        )

    async def close(self) -> None:
        return None


class RecordingHome:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def start(self, *args: object, **kwargs: object) -> RecordingHomeHandle:
        self.calls.append((args, kwargs))
        return RecordingHomeHandle()


class RecordingHomeHandle:
    async def result(self) -> HomeRunOutcome:
        return HomeRunOutcome(
            intent=HomeIntentType.CONTROL,
            status=HomeActionStatus.SUCCEEDED,
        )


class RecordingLiveSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.content_ranges: list[tuple[int, int]] = []
        self._played_offset = 0

    @property
    def played_offset(self) -> int:
        return self._played_offset

    async def start(self) -> None:
        self.calls.append(("start", None))

    def reset(self) -> None:
        self.calls.append(("reset", None))
        self._played_offset = 0

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        self.calls.append(("append", len(pcm)))
        self.content_ranges.append((start_offset, end_offset))
        self._played_offset = end_offset

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))

    async def wait_until_drained(self) -> None:
        return None


TTS_CONFIG = TtsConfiguration("fake-model", "fake-voice", 16_000)


class FakeProviderOutput:
    def __init__(self, sink: PromotablePcmSink) -> None:
        self.sink = sink
        self.configuration = TTS_CONFIG
        self.provider_state = TtsProviderState(
            ready=True,
            active=True,
            error_free=True,
            provider="fake",
            configuration=TTS_CONFIG,
        )
        self.calls: list[tuple[str, object | None]] = []
        self.played_offset = 0
        self.closed = asyncio.Event()

    def configure(self, *, model: str, voice: str) -> None:
        self.configuration = TtsConfiguration(model, voice, 16_000)

    async def start(self, *, text_offset: int = 0) -> None:
        self.played_offset = text_offset
        self.calls.append(("start", None))
        await self.sink.start()

    async def append(self, text: str) -> None:
        self.calls.append(("append", len(text)))
        await self.sink.append(
            b"\x01\x02" * max(1, len(text)),
            start_offset=0 if not self.calls[:-1] else self.played_offset,
            end_offset=self.played_offset + len(text),
        )
        self.played_offset += len(text)

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def stop(self) -> None:
        await self.interrupt()

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))

    async def pause(self) -> None:
        await self.sink.pause()

    async def resume(self) -> None:
        await self.sink.resume()

    async def duck(self) -> None:
        await self.sink.duck()

    async def unduck(self) -> None:
        await self.sink.unduck()

    async def close_transport(self) -> None:
        self.calls.append(("close_transport", None))
        self.closed.set()


class OrdinaryTtsOutput:
    def __init__(self, sink: RecordingLiveSink) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.played_offset = 0
        self.sink = sink

    async def start(self, *, text_offset: int = 0) -> None:
        self.played_offset = text_offset
        self.calls.append(("start", None))
        await self.sink.start()

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        start_offset = self.played_offset
        self.played_offset += len(text)
        await self.sink.append(
            b"\x03\x04" * max(1, len(text)),
            start_offset=start_offset,
            end_offset=self.played_offset,
        )

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def flush(self) -> None:
        # append() already delivers every sample; this fake has no text buffer.
        return None

    async def stop(self) -> None:
        self.calls.append(("stop", None))
        await self.sink.stop()

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))

    async def pause(self) -> None:
        return None

    async def resume(self) -> None:
        return None

    async def duck(self) -> None:
        return None

    async def unduck(self) -> None:
        return None


class ImmediateDecoder:
    def recognize(self, pcm: bytes) -> str:
        assert pcm
        return CONTENT


async def count_role(
    connection: asyncpg.Connection,
    session_id: UUID,
    role: str,
) -> int:
    value = await connection.fetchval(
        "SELECT count(*) FROM messages WHERE session_id = $1 AND role = $2",
        session_id,
        role,
    )
    return int(value)


@pytest.mark.asyncio
async def test_voice_draft_rollback_and_duplicate_commit_are_exactly_once(
    postgres_pool: asyncpg.Pool,
    tmp_path: Path,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute(
        "INSERT INTO fishes (id, snapshot_storage_version) VALUES ($1, 2)",
        FISH_ID,
    )
    await postgres_pool.execute(
        """
        INSERT INTO fish_sessions (id, fish_id, created_at)
        VALUES ($1, $2, $3)
        """,
        SESSION_ID,
        FISH_ID,
        datetime(2026, 8, 30, 4, tzinfo=UTC),
    )
    store = MessagesStore(postgres_pool)
    sessions = SessionStore(postgres_pool)
    llm = RecordingChatClient()
    home = RecordingHome()
    generation = ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )
    drafts = VoiceDraftCoordinator(
        prepare=_voice_draft_preparer(
            AnswerContextLoader(
                store,
                now=lambda: datetime(2026, 8, 30, 4, tzinfo=UTC),
                system_prompt="system",
            ),
            generation,
            enable_home=True,
        ),
        ttl_seconds=2,
        max_concurrency=2,
    )
    coordinator = AnswerCoordinator(
        store,
        llm,
        home_action_service=home,
        system_prompt="system",
    )
    app = FastAPI()
    app.state.message_store = store
    app.state.session_store = sessions
    app.state.answer_coordinator = coordinator
    app.state.voice_draft_coordinator = drafts
    app.include_router(catalog_router)
    app.include_router(internal_router)
    app.include_router(router)
    api_key = "fm_voice_draft_pg0"
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=ApiAuthService(
            ApiKeyCredentialStore(
                api_key=api_key,
                key_file=tmp_path / "api-key",
            )
        ),
    )
    public_events: list[str] = []

    async def consume_public_events() -> None:
        runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
        async for event in iter_answer_events(runtime):
            public_events.append(event.event)

    public_consumer = asyncio.create_task(consume_public_events())
    controller: VoiceDraftTurnController | None = None
    session: CompanionSession | None = None
    draft_tts: DraftTtsSession | None = None
    api: ApiCompanionClient | None = None
    try:
        async with _serve_app(app) as api_url:
            async with httpx.AsyncClient(base_url=api_url) as unauthenticated:
                rejected = await unauthenticated.get(
                    f"{BASE_PATH}/{ROLLBACK_TURN_ID}/events",
                    params={"revision": "1"},
                )
            assert rejected.status_code == 401
            api = ApiCompanionClient(api_url, api_key=api_key)
            live_sink = RecordingLiveSink()
            outputs: list[FakeProviderOutput] = []

            def output_factory(sink: PromotablePcmSink) -> FakeProviderOutput:
                output = FakeProviderOutput(sink)
                outputs.append(output)
                return output

            async def load_configuration() -> TtsConfiguration:
                return TTS_CONFIG

            draft_tts = DraftTtsSession(
                client=api,
                live_sink=live_sink,
                max_buffer_bytes=96_000,
                output_factory=output_factory,
                load_configuration=load_configuration,
            )
            ordinary_tts = OrdinaryTtsOutput(live_sink)
            session = CompanionSession(
                api,
                ordinary_tts,
                draft_tts=draft_tts,
            )
            await session.set_binding(FISH_ID)
            assert await session.on_wake_word(play_ack=False)
            controller = VoiceDraftTurnController(
                client=api,
                decoder=ImmediateDecoder(),
                draft_tts=draft_tts,
                turn_id_factory=iter((ROLLBACK_TURN_ID, TURN_ID)).__next__,
            )

            rollback_owner = controller.open_turn(
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                binding_generation=session.binding_generation,
                speech_generation=1,
            )
            rollback_candidate = controller.candidate(
                rollback_owner,
                pcm=b"\0" * 320,
            )
            assert rollback_candidate is not None
            assert await rollback_candidate is not None
            await _wait_until(
                lambda: bool(
                    outputs and any(call[0] == "append" for call in outputs[0].calls)
                )
            )

            assert live_sink.calls == []
            controller.abandon(rollback_owner, reason="rejected")
            await controller.wait_idle()
            await _wait_drafts_cleanup(drafts)
            await draft_tts.wait_idle()
            await asyncio.wait_for(outputs[0].closed.wait(), timeout=1)

            assert all(key.turn_id != ROLLBACK_TURN_ID for key in drafts._entries)
            assert draft_tts._key is None
            assert draft_tts._pending_key is None
            assert outputs[0].sink.buffered_bytes == 0
            assert ("interrupt", None) in outputs[0].calls
            assert live_sink.calls == []

            owner = controller.open_turn(
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                binding_generation=session.binding_generation,
                speech_generation=2,
            )
            candidate = controller.candidate(owner, pcm=b"\0" * 320)
            assert candidate is not None
            assert await candidate is not None
            await _wait_until(
                lambda: (
                    len(outputs) == 2
                    and any(call[0] == "append" for call in outputs[1].calls)
                )
            )

            async with postgres_pool.acquire() as connection:
                assert await count_role(connection, SESSION_ID, "user") == 0
                assert await count_role(connection, SESSION_ID, "assistant") == 0
                assert await connection.fetchval("SELECT count(*) FROM facts") == 0
                assert (
                    await connection.fetchval("SELECT count(*) FROM memory_snapshots")
                    == 0
                )
                assert (
                    await connection.fetchval("SELECT count(*) FROM home_action_runs")
                    == 0
                )
            assert home.calls == []
            assert public_events == []
            assert live_sink.calls == []
            submission = controller.commit(owner, content=CONTENT)
            assert await session.submit_voice_turn(
                submission.key.turn_id,
                submission.revision,
                submission.promote_revision,
                submission.content,
            )
            await asyncio.wait_for(session.wait_for_idle(), timeout=2)

            duplicate_result = await api.commit_voice_turn(
                submission.key,
                revision=submission.revision,
                promote_revision=submission.promote_revision,
                content=submission.content,
            )
            assert duplicate_result.message_id == TURN_ID
            assert duplicate_result.answer_expected is True

        await _wait_coordinator_idle(coordinator)
        runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
        assert runtime.llm_task is not None
        await asyncio.wait_for(asyncio.shield(runtime.llm_task), timeout=2)
        await asyncio.wait_for(public_consumer, timeout=2)

        async with postgres_pool.acquire() as connection:
            assert await count_role(connection, SESSION_ID, "user") == 1
            assert await count_role(connection, SESSION_ID, "assistant") == 1
            saved = await connection.fetchrow(
                "SELECT content, consumed_offset FROM messages "
                "WHERE session_id=$1 AND role='assistant'",
                SESSION_ID,
            )
            assert saved["consumed_offset"] == len(saved["content"]) > 0
        history = await store.list_messages(FISH_ID, SESSION_ID, limit=100)
        assert visible_message_content(history[-1]) == "已经打开。"
        assert runtime.playback_owner == "companion"
        assert len(home.calls) == 1
        assert public_events[0] == "answer.user"
        assert public_events[-1] == "answer.completed"
        assert live_sink.content_ranges, (
            outputs[1].calls,
            ordinary_tts.calls,
            public_events,
        )
        assert _ranges_do_not_overlap(live_sink.content_ranges)
        assert live_sink.content_ranges[0][0] == 0
        if ordinary_tts.calls:
            assert ordinary_tts.calls == [
                ("start", None),
                ("append", "已经打开。"),
                ("finish", None),
            ]
        else:
            assert [call for call in outputs[1].calls if call[0] == "append"] == [
                ("append", len("已经打开。"))
            ]
    finally:
        if not public_consumer.done():
            public_consumer.cancel()
        await asyncio.gather(public_consumer, return_exceptions=True)
        if controller is not None:
            await controller.close()
        if session is not None:
            await session.close()
        if draft_tts is not None:
            await draft_tts.close()
        if api is not None:
            await api.close()
        await drafts.close()
        await coordinator.close()


@pytest.mark.asyncio
async def test_task_voice_draft_has_zero_precommit_and_one_durable_run(
    postgres_pool: asyncpg.Pool,
    tmp_path: Path,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute(
        "INSERT INTO fishes (id, snapshot_storage_version) VALUES ($1, 2)",
        FISH_ID,
    )
    await postgres_pool.execute(
        """
        INSERT INTO fish_sessions (id, fish_id, created_at)
        VALUES ($1, $2, $3)
        """,
        SESSION_ID,
        FISH_ID,
        datetime(2026, 8, 30, 5, tzinfo=UTC),
    )
    messages = MessagesStore(postgres_pool)
    sessions = SessionStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    target = await targets.upsert_target(
        "hermes",
        "http://127.0.0.1:8642",
        "default",
        "",
        "hermes:test",
        {},
        "Hermes test",
    )
    policy = LocalConnectorCapabilityPolicy()
    route = AgentTaskRouteService(tasks)
    llm = RecordingChatClient()
    context_loader = AnswerContextLoader(
        messages,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=target.target.id,
        agent_capability_policy=policy,
        now=lambda: datetime(2026, 8, 30, 5, tzinfo=UTC),
        system_prompt="system",
    )
    generation = ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
        task_route_service=route,
    )
    drafts = VoiceDraftCoordinator(
        prepare=_voice_draft_preparer(
            context_loader,
            generation,
            enable_home=False,
        ),
        ttl_seconds=2,
        max_concurrency=2,
    )
    coordinator = AnswerCoordinator(
        messages,
        llm,
        system_prompt="system",
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=target.target.id,
        agent_capability_policy=policy,
        agent_task_route_service=route,
    )
    app = FastAPI()
    app.state.message_store = messages
    app.state.session_store = sessions
    app.state.answer_coordinator = coordinator
    app.state.voice_draft_coordinator = drafts
    app.include_router(internal_router)
    app.include_router(router)
    api_key = "fm_task_voice_draft_pg0"
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=ApiAuthService(
            ApiKeyCredentialStore(
                api_key=api_key,
                key_file=tmp_path / "task-api-key",
            )
        ),
    )
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    api = ApiCompanionClient(
        "http://test",
        http_client=http_client,
        api_key=api_key,
    )
    key = VoiceDraftKey(FISH_ID, SESSION_ID, TASK_TURN_ID)
    try:
        assert await api.upsert_voice_draft(
            key,
            revision=1,
            content=TASK_CONTENT,
        )
        await _wait_draft_ready(drafts, TASK_TURN_ID)

        precommit = await postgres_pool.fetchrow(
            """
            SELECT
                (SELECT count(*) FROM messages
                 WHERE session_id = $1 AND role = 'user') AS users,
                (SELECT count(*) FROM messages
                 WHERE session_id = $1 AND role = 'assistant') AS assistants,
                (SELECT count(*) FROM agent_tasks WHERE fish_id = $2) AS tasks,
                (SELECT count(*)
                 FROM agent_runs AS run
                 JOIN agent_tasks AS task ON task.id = run.task_id
                 WHERE task.fish_id = $2) AS runs
            """,
            SESSION_ID,
            FISH_ID,
        )
        assert precommit is not None
        assert dict(precommit) == {
            "users": 0,
            "assistants": 0,
            "tasks": 0,
            "runs": 0,
        }

        committed = await api.commit_voice_turn(
            key,
            revision=2,
            promote_revision=1,
            content=TASK_CONTENT,
        )
        duplicate = await api.commit_voice_turn(
            key,
            revision=2,
            promote_revision=1,
            content=TASK_CONTENT,
        )
        assert committed.message_id == duplicate.message_id == TASK_TURN_ID
        assert committed.answer_expected is True
        assert duplicate.answer_expected is True

        await _wait_coordinator_idle(coordinator)
        runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
        if runtime.llm_task is not None:
            await asyncio.wait_for(asyncio.shield(runtime.llm_task), timeout=2)

        counts = await postgres_pool.fetchrow(
            """
            SELECT
                (SELECT count(*) FROM messages
                 WHERE session_id = $1 AND role = 'user') AS users,
                (SELECT count(*) FROM messages
                 WHERE session_id = $1 AND role = 'assistant') AS assistants,
                (SELECT count(*) FROM agent_tasks WHERE fish_id = $2) AS tasks,
                (SELECT count(*)
                 FROM agent_runs AS run
                 JOIN agent_tasks AS task ON task.id = run.task_id
                 WHERE task.fish_id = $2) AS runs
            """,
            SESSION_ID,
            FISH_ID,
        )
        assert counts is not None
        assert dict(counts) == {
            "users": 1,
            "assistants": 1,
            "tasks": 1,
            "runs": 1,
        }
        run = await postgres_pool.fetchrow(
            """
            SELECT status, external_run_id
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            WHERE task.fish_id = $1
            """,
            FISH_ID,
        )
        assert run is not None
        assert run["status"] == "queued"
        assert run["external_run_id"] is None
    finally:
        await api.close()
        await http_client.aclose()
        await drafts.close()
        await coordinator.close()


async def _wait_draft_ready(
    drafts: VoiceDraftCoordinator,
    turn_id: UUID,
) -> None:
    entry = next(
        entry for key, entry in drafts._entries.items() if key.turn_id == turn_id
    )
    async with entry.condition:
        await asyncio.wait_for(
            entry.condition.wait_for(lambda: entry.ready or entry.state != "active"),
            timeout=1,
        )
    assert entry.ready, "voice draft terminated before becoming ready"


async def _wait_coordinator_idle(coordinator: AnswerCoordinator) -> None:
    async def wait() -> None:
        while coordinator._background_tasks:
            await asyncio.gather(
                *tuple(coordinator._background_tasks),
                return_exceptions=False,
            )

    await asyncio.wait_for(wait(), timeout=2)


async def _wait_drafts_cleanup(drafts: VoiceDraftCoordinator) -> None:
    async def wait() -> None:
        while drafts._entries or drafts._cleanup_tasks:  # noqa: ASYNC110
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


def _ranges_do_not_overlap(ranges: Sequence[tuple[int, int]]) -> bool:
    previous_end = 0
    for start, end in ranges:
        if start < previous_end or end < start:
            return False
        previous_end = max(previous_end, end)
    return True


@asynccontextmanager
async def _serve_app(app: FastAPI) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
    )
    task = asyncio.create_task(
        server.serve(sockets=[listener]),
        name="voice-draft-pg0-http-server",
    )

    async def wait_started() -> None:
        while not server.started:  # noqa: ASYNC110
            await asyncio.sleep(0.001)

    try:
        await asyncio.wait_for(wait_started(), timeout=2)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=2)
        finally:
            listener.close()
