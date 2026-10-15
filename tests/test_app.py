import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import fish_memory.app as app_module
from fish_memory.agent.connector import AgentConnectorRuntime, ConnectorDescriptor
from fish_memory.config import Settings
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import TaskStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.llm.runtime import ModelRoute

EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)
AUTH_HEADERS = {"Authorization": "Bearer fm_test_service_key"}


def underlying_model(client):
    return (
        client._owner.client(client._name) if isinstance(client, ModelRoute) else client
    )


@pytest.fixture(autouse=True)
def stub_proactive_startup_database_recovery(monkeypatch, tmp_path):
    # This module stubs database I/O; real recovery is covered by PostgreSQL tests.
    # Settings defaults must never inspect the developer's profile or model cache.
    monkeypatch.chdir(tmp_path)

    async def recover(_store, **_kwargs):
        return 0

    monkeypatch.setattr(
        app_module.BackgroundMessageStore, "cancel_orphaned_proactive_streams", recover
    )
    monkeypatch.setattr(app_module.ExtensionStore, "recover", recover)
    monkeypatch.setattr(app_module.LlmLogStore, "recover", recover)

    async def no_active_runs(_store):
        return ()

    monkeypatch.setattr(app_module.AutonomousTaskStore, "active", no_active_runs)
    monkeypatch.setattr(app_module.PluginTaskStore, "active", no_active_runs)
    monkeypatch.setenv("FISH_PLUGIN_ROOT", str(tmp_path / "plugins"))


def test_recall_context_token_budget_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="postgresql://test",
            llm_provider="openai",
            llm_api_key=SecretStr("key"),
            llm_model="qwen-plus",
            llm_base_url="https://example.invalid/v1",
            FISH_MEMORY_RECALL_CONTEXT_TOKEN_BUDGET=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["", "当前活动：主持直播"])
async def test_voice_draft_preparer_forwards_agent_route_context(activity: str) -> None:
    fish_id = UUID(int=101)
    session_id = UUID(int=102)
    turn_id = UUID(int=103)
    created_at = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    content = "继续刚才的任务"
    messages = [SimpleNamespace(role="user", content=content)]
    recall_conversation = (SimpleNamespace(role="assistant"),)
    agent_target = SimpleNamespace(target=object())
    task_lane_context = SimpleNamespace(candidates=(object(),))
    prepared = object()
    captured: dict[str, object] = {}

    class FakeContextLoader:
        async def load_for_draft(
            self,
            actual_fish_id: UUID,
            actual_session_id: UUID,
            *,
            turn_id: UUID,
            content: str,
            created_at: datetime,
            tracker: object,
        ) -> SimpleNamespace:
            captured["loader_args"] = (
                actual_fish_id,
                actual_session_id,
                turn_id,
                content,
                created_at,
            )
            captured["loader_tracker"] = tracker
            return SimpleNamespace(
                messages=messages,
                recall_conversation=recall_conversation,
                agent_target=agent_target,
                task_lane_context=task_lane_context,
                activity_context=activity,
            )

    class FakeGeneration:
        def prepare_initial(
            self,
            actual_fish_id: UUID,
            actual_messages: object,
            actual_recall_conversation: object,
            sink: object,
            tracker: object,
            **kwargs: object,
        ) -> object:
            captured["generation_args"] = (
                actual_fish_id,
                actual_messages,
                actual_recall_conversation,
                sink,
                tracker,
            )
            captured["generation_kwargs"] = kwargs
            return prepared

    prepare = app_module._voice_draft_preparer(
        cast(Any, FakeContextLoader()),
        cast(Any, FakeGeneration()),
        enable_home=False,
    )

    result = await prepare(
        app_module.VoiceDraftKey(fish_id, session_id, turn_id),
        7,
        content,
        created_at=created_at,
    )

    assert result is prepared
    assert captured["loader_args"] == (
        fish_id,
        session_id,
        turn_id,
        content,
        created_at,
    )
    generation_args = cast(tuple[object, ...], captured["generation_args"])
    assert generation_args[0] == fish_id
    assert generation_args[1] is messages
    assert generation_args[2] is recall_conversation
    assert generation_args[4] is captured["loader_tracker"]
    generation_kwargs = cast(dict[str, object], captured["generation_kwargs"])
    assert generation_kwargs["enable_home"] is False
    assert generation_kwargs["agent_target"] is agent_target
    assert generation_kwargs["task_lane_context"] is task_lane_context
    assert generation_kwargs.get("activity_context", "") == activity


def test_app_registers_kws_routes() -> None:
    settings = Settings.model_construct(
        database_url="postgresql://test",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
    )

    app = app_module.create_app(settings)
    paths = {
        route.path
        for included in app.routes
        if hasattr(included, "original_router")
        for route in included.original_router.routes
    }
    paths.update(route.path for route in app.routes if hasattr(route, "path"))

    assert "/v1/kws/status" in paths
    assert "/v1/kws/stream" in paths
    assert "/kws/status" not in paths
    assert "/kws/stream" not in paths


@pytest.mark.asyncio
async def test_health_ready_and_legacy_routes_have_explicit_boundaries() -> None:
    settings = Settings.model_construct(
        database_url="postgresql://test",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
    )
    app = app_module.create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        ready = await client.get("/ready")
        capabilities_unauthenticated = await client.get("/internal/capabilities")
        capabilities = await client.get("/internal/capabilities", headers=AUTH_HEADERS)
        legacy = [
            await client.get("/fishes", headers=AUTH_HEADERS),
            await client.get("/tts/models", headers=AUTH_HEADERS),
            await client.get("/kws/status", headers=AUTH_HEADERS),
            await client.get("/v1/tts/custom-voices", headers=AUTH_HEADERS),
        ]
        openapi = await client.get("/openapi.json", headers=AUTH_HEADERS)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json() == {"status": "not_ready"}
    assert capabilities_unauthenticated.status_code == 401
    assert capabilities.status_code == 200
    assert capabilities.json() == {"voiceDrafts": True}
    assert [response.status_code for response in legacy] == [404] * 4
    assert all(not path.startswith("/internal/") for path in openapi.json()["paths"])


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_models", [False, True])
async def test_lifespan_initializes_and_closes_services_in_order(
    monkeypatch: pytest.MonkeyPatch,
    separate_models: bool,
) -> None:
    events: list[str] = []
    llm_instances: list[object] = []
    embedding_instances: list[object] = []
    processor_semaphores: dict[str, asyncio.Semaphore] = {}

    class FakeDatabase:
        database_url = "postgresql://resolved"

        async def close(self) -> None:
            events.append("database.close")

    async def fake_resolve_database(
        database_url: str,
        *,
        data_dir: str | None = None,
        desktop_mode: bool = False,
    ) -> FakeDatabase:
        assert database_url == "pg0://fish-memory"
        assert data_dir is None
        assert desktop_mode is False
        events.append("database.resolve")
        return FakeDatabase()

    class FakePool:
        async def close(self) -> None:
            events.append("pool.close")

    pool = FakePool()

    async def fake_create_pool(
        database_url: str, *, desktop_mode: bool = False
    ) -> FakePool:
        assert database_url == "postgresql://resolved"
        assert desktop_mode is False
        events.append("pool.create")
        return pool

    async def fake_run_migrations(value: object) -> None:
        assert value is pool
        events.append("migrate")

    class FakeStore:
        def __init__(self, value: object, **kwargs: object) -> None:
            assert value is pool
            assert callable(kwargs["now"])
            assert kwargs["embedding_space"] == EMBEDDING_SPACE
            events.append("store.create")

        async def fail_stale_generating_answers(self) -> int:
            events.append("recover")
            return 0

    class FakePerformanceStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("performance_store.create")

    class FakeLlm:
        def __init__(self, **kwargs: object) -> None:
            assert isinstance(kwargs.pop("call_journal"), app_module.CallJournal)
            background = separate_models and bool(llm_instances)
            assert kwargs == {
                "api_key": "background-key" if background else "key",
                "base_url": "https://background.invalid/v1"
                if background
                else "https://example.invalid/v1",
                "model": "background-model" if background else "qwen-plus",
            }
            llm_instances.append(self)
            events.append("llm.create")

        async def close(self) -> None:
            events.append("llm.close")

    class FakeEmbedding:
        space = EMBEDDING_SPACE

        def __init__(self) -> None:
            embedding_instances.append(self)
            events.append("embedding.create")

        async def initialize(self) -> None:
            events.append("embedding.initialize")

        async def close(self) -> None:
            events.append("embedding.close")

    def fake_create_embedding_client(settings: Settings) -> FakeEmbedding:
        assert settings.embedding_model == "text-embedding-v4"
        assert settings.embedding_dimensions == 1024
        return FakeEmbedding()

    async def fake_ensure_embedding_hnsw_index(
        value: object,
        dimensions: int,
    ) -> None:
        assert value is pool
        assert dimensions == 1024
        events.append("embedding.index")

    class FakeEmbeddingSpaceStore:
        def __init__(self, value: object) -> None:
            assert value is pool

    class FakeCoordinator:
        def __init__(
            self,
            store: object,
            llm: object,
            *,
            speech_tags_provider: object,
            experiment: object,
            message_attachments: object,
            session_history: object,
            performance_store: object,
            fish_store: object,
            snapshot_store: object,
            recall_service: object,
            rquery_integrity_service: object,
            recall_context_token_budget: int,
            answer_history_message_limit: int,
            dynamic_snapshot_max_age: timedelta,
            now: object,
            system_prompt: str,
            system_prompt_extension: object,
            activity_context_provider: object,
            answer_enable_search: bool,
            home_action_service: object,
            environment_context_loader: object,
            on_ask_submitted: object,
        ) -> None:
            assert isinstance(store, FakeStore)
            assert callable(speech_tags_provider)
            assert isinstance(experiment, app_module.ExperimentalService)
            assert isinstance(message_attachments, app_module.MessageAttachmentStore)
            assert isinstance(session_history, app_module.SessionHistoryService)
            assert underlying_model(llm) is llm_instances[0]
            assert isinstance(performance_store, FakePerformanceStore)
            assert fish_store is not None
            assert isinstance(snapshot_store, FakeSnapshotStore)
            assert isinstance(recall_service, FakeRecallService)
            assert isinstance(
                rquery_integrity_service,
                app_module.RQueryIntegrityService,
            )
            assert recall_context_token_budget == 12_345
            assert answer_history_message_limit == 73
            assert dynamic_snapshot_max_age == timedelta(days=9)
            assert callable(now)
            assert system_prompt == "system"
            assert system_prompt_extension is None
            assert callable(activity_context_provider)
            assert answer_enable_search is True
            assert home_action_service is None
            assert isinstance(
                environment_context_loader,
                app_module.EnvironmentAnswerContextLoader,
            )
            assert callable(on_ask_submitted)
            events.append("coordinator.create")

        def has_active_generation(self) -> bool:
            return False

        async def close(self) -> None:
            events.append("coordinator.close")

    class FakeVoiceDraftCoordinator:
        def __init__(
            self,
            *,
            prepare: object,
            ttl_seconds: float,
            max_concurrency: int,
            now: object,
        ) -> None:
            assert callable(prepare)
            assert ttl_seconds == 2.0
            assert max_concurrency == 2
            assert callable(now)
            events.append("voice_drafts.create")

        async def close(self) -> None:
            events.append("voice_drafts.close")

    class FakeFactStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("fact_store.create")

    class FakeSnapshotStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("snapshot_store.create")

    class FakeRecallStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("recall_store.create")

    class FakeEntityStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("entity_store.create")

    class FakeRecallPlanner:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            events.append("recall_planner.create")

    class FakeRecallService:
        def __init__(
            self,
            store: object,
            planner: object,
            embedder: object,
            embedding_space_store: object,
            *,
            now: object,
        ) -> None:
            assert isinstance(store, FakeRecallStore)
            assert isinstance(planner, FakeRecallPlanner)
            assert embedder is embedding_instances[0]
            assert isinstance(embedding_space_store, FakeEmbeddingSpaceStore)
            assert callable(now)
            events.append("recall_service.create")

    class FakeFactExtractor:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            events.append("fact_extractor.create")

    class FakeFactProcessor:
        def __init__(
            self,
            store: object,
            extractor: object,
            embedder: object,
            performance_store: object,
            *,
            embedding_space_store: object,
            llm_semaphore: asyncio.Semaphore,
            embedding_semaphore: asyncio.Semaphore,
            max_retries: int,
            batch_size: int,
            llm_timeout_seconds: float,
            now: object,
        ) -> None:
            assert isinstance(store, FakeFactStore)
            assert isinstance(extractor, FakeFactExtractor)
            assert embedder is embedding_instances[0]
            assert isinstance(performance_store, FakePerformanceStore)
            assert isinstance(embedding_space_store, FakeEmbeddingSpaceStore)
            assert llm_semaphore._value == 2
            assert embedding_semaphore._value == 3
            assert llm_semaphore is not embedding_semaphore
            assert max_retries == 4
            assert batch_size == 12
            assert llm_timeout_seconds == 120
            assert callable(now)
            processor_semaphores["fact"] = llm_semaphore
            processor_semaphores["fact_embedding"] = embedding_semaphore
            events.append("fact_processor.create")

    class FakeSnapshotUpdater:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            events.append("snapshot_updater.create")

    class FakeStaticProfileUpdater:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            events.append("static_updater.create")

    confirmer_instances: list[object] = []
    static_confirmer_instances: list[object] = []

    class FakeSnapshotUpdateConfirmer:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            confirmer_instances.append(self)
            events.append("snapshot_confirmer.create")

    class FakeStaticProfileUpdateConfirmer:
        def __init__(self, client: object) -> None:
            assert underlying_model(client) is llm_instances[-1]
            static_confirmer_instances.append(self)
            events.append("static_confirmer.create")

    class FakeSnapshotProcessor:
        def __init__(
            self,
            store: object,
            updater: object,
            confirmer: object,
            resolver: object,
            entity_store: object,
            performance_store: object,
            *,
            llm_semaphore: asyncio.Semaphore,
            max_retries: int,
            batch_size: int,
            now: object,
        ) -> None:
            assert isinstance(store, FakeSnapshotStore)
            assert isinstance(updater, FakeSnapshotUpdater)
            assert confirmer is confirmer_instances[0]
            assert isinstance(resolver, app_module.EntityResolver)
            assert isinstance(entity_store, FakeEntityStore)
            assert isinstance(performance_store, FakePerformanceStore)
            assert llm_semaphore._value == 2
            assert max_retries == 4
            assert batch_size == 50
            assert callable(now)
            processor_semaphores["dynamic"] = llm_semaphore
            events.append("snapshot_processor.create")

    class FakeStaticProfileProcessor:
        def __init__(
            self,
            store: object,
            updater: object,
            confirmer: object,
            resolver: object,
            entity_store: object,
            performance_store: object,
            *,
            llm_semaphore: asyncio.Semaphore,
            max_retries: int,
            batch_size: int,
            now: object,
        ) -> None:
            assert isinstance(store, FakeSnapshotStore)
            assert isinstance(updater, FakeStaticProfileUpdater)
            assert confirmer is static_confirmer_instances[0]
            assert confirmer is not confirmer_instances[0]
            assert isinstance(resolver, app_module.EntityResolver)
            assert isinstance(entity_store, FakeEntityStore)
            assert isinstance(performance_store, FakePerformanceStore)
            assert llm_semaphore._value == 2
            assert max_retries == 4
            assert batch_size == 50
            assert callable(now)
            processor_semaphores["static"] = llm_semaphore
            events.append("static_processor.create")

    class FakeSnapshotItemRelationStore:
        def __init__(
            self,
            value: object,
            *,
            snapshot_type: str,
        ) -> None:
            assert value is pool
            assert snapshot_type in {"dynamic", "static"}
            events.append(f"relation_store.create:{snapshot_type}")

    class FakeSnapshotItemRelationProcessor:
        def __init__(
            self,
            store: object,
            embedder: object,
            *,
            embedding_space_store: object,
            embedding_semaphore: asyncio.Semaphore,
            snapshot_type: str,
            max_retries: int,
            batch_size: int,
        ) -> None:
            assert isinstance(store, FakeSnapshotItemRelationStore)
            assert embedder is embedding_instances[0]
            assert isinstance(embedding_space_store, FakeEmbeddingSpaceStore)
            assert embedding_semaphore._value == 3
            assert snapshot_type in {"dynamic", "static"}
            assert max_retries == 4
            assert batch_size == 50
            processor_semaphores[f"relation_embedding_{snapshot_type}"] = (
                embedding_semaphore
            )
            events.append(f"relation_processor.create:{snapshot_type}")

    class FakeMaskProcessor:
        def __init__(
            self,
            store: object,
            embedder: object,
            *,
            batch_size: int,
            embedding_semaphore: asyncio.Semaphore,
        ) -> None:
            assert isinstance(store, FakeRecallStore)
            assert embedder is embedding_instances[0]
            assert batch_size == 50
            assert embedding_semaphore is processor_semaphores["fact_embedding"]
            events.append("mask_processor.create")

    class FakeSupervisor:
        def __init__(
            self,
            processor: object,
            *,
            name: str,
            poll_interval_seconds: float,
        ) -> None:
            assert isinstance(
                processor,
                (
                    FakeFactProcessor,
                    FakeSnapshotProcessor,
                    FakeStaticProfileProcessor,
                    FakeSnapshotItemRelationProcessor,
                    FakeMaskProcessor,
                    app_module.BackgroundMessageExpiryProcessor,
                    app_module.LlmLogStore,
                ),
            )
            assert name in {
                "llm-log-expiry",
                "background-message-expiry-processor",
                "fact-processor",
                "dynamic-snapshot-processor",
                "static-profile-processor",
                "dynamic-relation-processor",
                "static-relation-processor",
                "mask-embedding-processor",
            }
            assert poll_interval_seconds == (
                app_module.LOG_CLEANUP_INTERVAL_SECONDS
                if name == "llm-log-expiry"
                else 0.25
            )
            events.append(f"supervisor.create:{name}")

        def start(self) -> None:
            events.append("supervisor.start")

        async def close(self) -> None:
            events.append("supervisor.close")

    monkeypatch.setattr(
        app_module,
        "resolve_database",
        fake_resolve_database,
        raising=False,
    )
    monkeypatch.setattr(app_module, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(app_module, "MessagesStore", FakeStore)
    monkeypatch.setattr(app_module, "PerformanceStore", FakePerformanceStore)
    monkeypatch.setattr(app_module, "BailianChatClient", FakeLlm)
    monkeypatch.setattr(
        app_module,
        "create_embedding_client",
        fake_create_embedding_client,
    )
    monkeypatch.setattr(
        app_module,
        "ensure_embedding_hnsw_index",
        fake_ensure_embedding_hnsw_index,
    )
    monkeypatch.setattr(
        app_module,
        "FishEmbeddingSpaceStore",
        FakeEmbeddingSpaceStore,
    )
    monkeypatch.setattr(app_module, "AnswerCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        app_module,
        "VoiceDraftCoordinator",
        FakeVoiceDraftCoordinator,
        raising=False,
    )
    monkeypatch.setattr(app_module, "FactStore", FakeFactStore, raising=False)
    monkeypatch.setattr(
        app_module,
        "SnapshotStore",
        FakeSnapshotStore,
        raising=False,
    )
    monkeypatch.setattr(app_module, "RecallStore", FakeRecallStore)
    monkeypatch.setattr(app_module, "EntityStore", FakeEntityStore)
    monkeypatch.setattr(app_module, "RecallPlanner", FakeRecallPlanner)
    monkeypatch.setattr(app_module, "FactRecallService", FakeRecallService)
    monkeypatch.setattr(
        app_module,
        "FactExtractor",
        FakeFactExtractor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "FactProcessor",
        FakeFactProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "DynamicSnapshotUpdater",
        FakeSnapshotUpdater,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "DynamicSnapshotProcessor",
        FakeSnapshotProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "StaticProfileUpdater",
        FakeStaticProfileUpdater,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "SnapshotUpdateConfirmer",
        FakeSnapshotUpdateConfirmer,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "StaticProfileUpdateConfirmer",
        FakeStaticProfileUpdateConfirmer,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "StaticProfileProcessor",
        FakeStaticProfileProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "SnapshotItemRelationStore",
        FakeSnapshotItemRelationStore,
    )
    monkeypatch.setattr(
        app_module,
        "SnapshotItemRelationProcessor",
        FakeSnapshotItemRelationProcessor,
    )
    monkeypatch.setattr(
        app_module,
        "MaskEmbeddingProcessor",
        FakeMaskProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "BackgroundTaskSupervisor",
        FakeSupervisor,
        raising=False,
    )
    settings = Settings.model_construct(
        database_url="pg0://fish-memory",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
        answer_enable_search=True,
        embedding_model="text-embedding-v4",
        embedding_dimensions=1024,
        background_poll_interval_seconds=0.25,
        background_llm_concurrency=2,
        background_embedding_concurrency=3,
        fact_extraction_max_retries=4,
        fact_batch_size=12,
        dynamic_snapshot_max_age_days=9,
        snapshot_batch_size=50,
        recall_context_token_budget=12_345,
        answer_history_message_limit=73,
        companion_voice_draft_enabled=True,
        companion_voice_draft_ttl_ms=2000,
        companion_voice_draft_max_concurrency=2,
    )

    if separate_models:
        from fish_memory.llm.configuration import DesktopModelRoutes

        settings.desktop_model_routes = DesktopModelRoutes.model_validate(
            {
                "chat": {
                    "baseUrl": "https://example.invalid/v1",
                    "model": "qwen-plus",
                    "apiKey": "key",
                },
                "reasoning": {
                    "baseUrl": "https://background.invalid/v1",
                    "model": "background-model",
                    "apiKey": "background-key",
                },
            }
        )
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/openapi.json", headers=AUTH_HEADERS)
        assert response.status_code == 200
        assert isinstance(app.state.fact_store, FakeFactStore)
        assert isinstance(app.state.snapshot_store, FakeSnapshotStore)
        assert isinstance(app.state.entity_store, FakeEntityStore)
        assert app.state.pool is pool
        assert underlying_model(app.state.llm) is llm_instances[0]
        assert isinstance(
            app.state.voice_draft_coordinator,
            FakeVoiceDraftCoordinator,
        )
        assert len(llm_instances) == (2 if separate_models else 1)
        assert underlying_model(app.state.reasoning_llm) is llm_instances[-1]
        proactive = app.state.proactive_coordinator
        experiment = app.state.experimental_companion
        assert proactive.experimental.service is experiment
        assert experiment.chat is app.state.llm
        assert experiment.reasoning is app.state.reasoning_llm
        history = app.state.environment_history_service._summarizer
        assert history._model == (
            "background-model" if separate_models else "qwen-plus"
        )
        assert isinstance(history, ModelRoute)
        assert history._owner is app.state.llm._owner
        assert len(confirmer_instances) == 1
        assert len(static_confirmer_instances) == 1
        assert processor_semaphores["fact"] is not processor_semaphores["dynamic"]
        assert (
            processor_semaphores["fact_embedding"] is not processor_semaphores["fact"]
        )
        assert processor_semaphores["dynamic"] is not processor_semaphores["static"]
        assert processor_semaphores["fact"] is not processor_semaphores["static"]
        assert (
            processor_semaphores["relation_embedding_dynamic"]
            is processor_semaphores["fact_embedding"]
        )
        assert (
            processor_semaphores["relation_embedding_static"]
            is processor_semaphores["fact_embedding"]
        )
        assert events == [
            "database.resolve",
            "pool.create",
            "migrate",
            "embedding.create",
            "embedding.initialize",
            "embedding.index",
            "store.create",
            "recover",
            "performance_store.create",
            "llm.create",
            *(["llm.create"] if separate_models else []),
            "snapshot_store.create",
            "entity_store.create",
            "recall_store.create",
            "recall_planner.create",
            "recall_service.create",
            "coordinator.create",
            "voice_drafts.create",
            "fact_store.create",
            "fact_extractor.create",
            "fact_processor.create",
            "snapshot_updater.create",
            "snapshot_confirmer.create",
            "snapshot_processor.create",
            "static_updater.create",
            "static_confirmer.create",
            "static_processor.create",
            "relation_store.create:dynamic",
            "relation_processor.create:dynamic",
            "relation_store.create:static",
            "relation_processor.create:static",
            "mask_processor.create",
            "supervisor.create:llm-log-expiry",
            "supervisor.create:background-message-expiry-processor",
            "supervisor.create:fact-processor",
            "supervisor.create:dynamic-snapshot-processor",
            "supervisor.create:static-profile-processor",
            "supervisor.create:dynamic-relation-processor",
            "supervisor.create:static-relation-processor",
            "supervisor.create:mask-embedding-processor",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
            "supervisor.start",
        ]

    assert events[-(15 if separate_models else 14) :] == [
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "voice_drafts.close",
        "coordinator.close",
        "llm.close",
        *(["llm.close"] if separate_models else []),
        "embedding.close",
        "pool.close",
        "database.close",
    ]


@pytest.mark.asyncio
async def test_lifespan_cleans_up_created_resources_when_supervisor_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeDatabase:
        database_url = "postgresql://resolved"

        async def close(self) -> None:
            events.append("database.close")

    async def fake_resolve_database(
        database_url: str,
        *,
        data_dir: str | None = None,
        desktop_mode: bool = False,
    ) -> FakeDatabase:
        assert database_url == "pg0://fish-memory"
        assert data_dir is None
        assert desktop_mode is False
        events.append("database.resolve")
        return FakeDatabase()

    class FakePool:
        async def close(self) -> None:
            events.append("pool.close")

    pool = FakePool()

    async def fake_create_pool(
        database_url: str, *, desktop_mode: bool = False
    ) -> FakePool:
        assert database_url == "postgresql://resolved"
        assert desktop_mode is False
        events.append("pool.create")
        return pool

    async def fake_run_migrations(value: object) -> None:
        assert value is pool
        events.append("migrate")

    class FakeStore:
        def __init__(self, value: object, **kwargs: object) -> None:
            assert value is pool
            assert callable(kwargs["now"])
            events.append("store.create")

        async def fail_stale_generating_answers(self) -> int:
            events.append("recover")
            return 0

    class FakePerformanceStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("performance_store.create")

    class FakeLlm:
        def __init__(self, **kwargs: str) -> None:
            del kwargs
            events.append("llm.create")

        async def close(self) -> None:
            events.append("llm.close")

    class FakeEmbedding:
        space = EMBEDDING_SPACE

        def __init__(self) -> None:
            events.append("embedding.create")

        async def initialize(self) -> None:
            events.append("embedding.initialize")

        async def close(self) -> None:
            events.append("embedding.close")

    async def fake_ensure_embedding_hnsw_index(
        value: object,
        dimensions: int,
    ) -> None:
        assert value is pool
        assert dimensions == 1024
        events.append("embedding.index")

    class FakeCoordinator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("coordinator.create")

        def has_active_generation(self) -> bool:
            return False

        async def close(self) -> None:
            events.append("coordinator.close")

    class FakeFactStore:
        def __init__(self, value: object) -> None:
            assert value is pool
            events.append("fact_store.create")

    class FakeFactExtractor:
        def __init__(self, client: object) -> None:
            assert isinstance(underlying_model(client), FakeLlm)
            events.append("fact_extractor.create")

    class FakeFactProcessor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("fact_processor.create")

    class FailingSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("supervisor.create")

        def start(self) -> None:
            events.append("supervisor.start")
            raise RuntimeError("supervisor failed")

        async def close(self) -> None:
            events.append("supervisor.close")

    monkeypatch.setattr(
        app_module,
        "resolve_database",
        fake_resolve_database,
        raising=False,
    )
    monkeypatch.setattr(app_module, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(app_module, "MessagesStore", FakeStore)
    monkeypatch.setattr(app_module, "PerformanceStore", FakePerformanceStore)
    monkeypatch.setattr(app_module, "BailianChatClient", FakeLlm)
    monkeypatch.setattr(
        app_module,
        "create_embedding_client",
        lambda settings: FakeEmbedding(),
    )
    monkeypatch.setattr(
        app_module,
        "ensure_embedding_hnsw_index",
        fake_ensure_embedding_hnsw_index,
    )
    monkeypatch.setattr(app_module, "AnswerCoordinator", FakeCoordinator)
    monkeypatch.setattr(app_module, "FactStore", FakeFactStore, raising=False)
    monkeypatch.setattr(
        app_module,
        "FactExtractor",
        FakeFactExtractor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "FactProcessor",
        FakeFactProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "BackgroundTaskSupervisor",
        FailingSupervisor,
        raising=False,
    )
    settings = Settings.model_construct(
        database_url="pg0://fish-memory",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
        embedding_model="text-embedding-v4",
        embedding_dimensions=1024,
        background_poll_interval_seconds=0.25,
        background_llm_concurrency=2,
        fact_extraction_max_retries=4,
    )

    app = app_module.create_app(settings)
    with pytest.raises(RuntimeError, match="supervisor failed"):
        async with app.router.lifespan_context(app):
            pass

    assert events == [
        "database.resolve",
        "pool.create",
        "migrate",
        "embedding.create",
        "embedding.initialize",
        "embedding.index",
        "store.create",
        "recover",
        "performance_store.create",
        "llm.create",
        "coordinator.create",
        "fact_store.create",
        "fact_extractor.create",
        "fact_processor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.create",
        "supervisor.start",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "supervisor.close",
        "coordinator.close",
        "llm.close",
        "embedding.close",
        "pool.close",
        "database.close",
    ]


@pytest.mark.asyncio
async def test_lifespan_closes_database_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeDatabase:
        database_url = "postgresql://resolved"

        async def close(self) -> None:
            events.append("database.close")

    class FakePool:
        async def close(self) -> None:
            events.append("pool.close")

    async def fake_resolve_database(
        database_url: str,
        *,
        data_dir: str | None = None,
        desktop_mode: bool = False,
    ) -> FakeDatabase:
        assert database_url == "pg0://fish-memory"
        assert data_dir is None
        assert desktop_mode is False
        events.append("database.resolve")
        return FakeDatabase()

    async def fake_create_pool(
        database_url: str, *, desktop_mode: bool = False
    ) -> FakePool:
        assert database_url == "postgresql://resolved"
        assert desktop_mode is False
        events.append("pool.create")
        return FakePool()

    async def fail_migration(pool: object) -> None:
        del pool
        events.append("migrate")
        raise RuntimeError("migration failed")

    monkeypatch.setattr(
        app_module,
        "resolve_database",
        fake_resolve_database,
        raising=False,
    )
    monkeypatch.setattr(app_module, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "run_migrations", fail_migration)
    settings = Settings.model_construct(
        database_url="pg0://fish-memory",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
    )

    app = app_module.create_app(settings)
    with pytest.raises(RuntimeError, match="migration failed"):
        async with app.router.lifespan_context(app):
            pass

    assert events == [
        "database.resolve",
        "pool.create",
        "migrate",
        "pool.close",
        "database.close",
    ]


@pytest.mark.asyncio
async def test_validation_errors_use_stable_error_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoopPool:
        async def close(self) -> None:
            return None

    async def fake_create_pool(
        database_url: str, *, desktop_mode: bool = False
    ) -> NoopPool:
        del database_url, desktop_mode
        return NoopPool()

    async def fake_run_migrations(pool: object) -> None:
        del pool

    class Store:
        def __init__(self, pool: object, **kwargs: object) -> None:
            del pool
            assert callable(kwargs["now"])

        async def fail_stale_generating_answers(self) -> int:
            return 0

    class Resource:
        space = EMBEDDING_SPACE

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            return None

        def has_active_generation(self) -> bool:
            return False

        async def close(self) -> None:
            return None

        async def initialize(self) -> None:
            return None

    monkeypatch.setattr(app_module, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(app_module, "MessagesStore", Store)
    monkeypatch.setattr(app_module, "BailianChatClient", Resource)
    monkeypatch.setattr(
        app_module,
        "create_embedding_client",
        lambda settings: Resource(),
    )
    monkeypatch.setattr(
        app_module,
        "ensure_embedding_hnsw_index",
        lambda pool, dimensions: asyncio.sleep(0),
    )
    monkeypatch.setattr(app_module, "AnswerCoordinator", Resource)
    monkeypatch.setattr(app_module, "FactStore", Resource, raising=False)
    monkeypatch.setattr(app_module, "FactExtractor", Resource, raising=False)
    monkeypatch.setattr(app_module, "FactProcessor", Resource, raising=False)
    monkeypatch.setattr(
        app_module,
        "BackgroundTaskSupervisor",
        Resource,
        raising=False,
    )
    settings = Settings.model_construct(
        database_url="postgresql://test",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
        embedding_model="text-embedding-v4",
        embedding_dimensions=1024,
        background_poll_interval_seconds=0.25,
        background_llm_concurrency=2,
        fact_extraction_max_retries=4,
    )

    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/fishes/not-a-uuid/ask",
                json={"content": 123},
                headers=AUTH_HEADERS,
            )

    assert response.status_code == 400
    assert set(response.json()["error"]) == {"code", "message"}


def _agent_runtime_settings(*, enabled: bool) -> Settings:
    return Settings.model_construct(
        database_url="postgresql://test",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr("fm_test_service_key"),
        desktop_mode=True,
        agent_tasks_enabled=enabled,
        agent_url="http://127.0.0.1:8642/p/coder",
        agent_api_key=SecretStr("provider-key"),
        agent_credentials_file=".cache/fish-memory/test-agent-credentials.json",
        agent_run_concurrency=3,
        agent_run_poll_interval_seconds=0.125,
        agent_run_stale_timeout_seconds=19.0,
        agent_http_timeout_seconds=7.0,
        agent_sse_read_timeout_seconds=11.0,
        background_poll_interval_seconds=0.25,
        background_llm_concurrency=2,
        companion_voice_draft_enabled=True,
        companion_voice_draft_ttl_ms=2000,
        companion_voice_draft_max_concurrency=2,
    )


def _install_agent_runtime_lifespan_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, object]]:
    events: list[str] = []
    captured: dict[str, object] = {"foreground_active": False}

    class FakeDatabase:
        database_url = "postgresql://resolved"

        async def close(self) -> None:
            events.append("database.close")

    async def fake_resolve_database(
        database_url: str,
        *,
        data_dir: str | None = None,
        desktop_mode: bool = False,
    ) -> FakeDatabase:
        del database_url, data_dir
        assert desktop_mode is True
        return FakeDatabase()

    class FakePool:
        async def fetch(self, query: str) -> list[object]:
            assert "run.origin = 'proactive'" in query
            return []

        async def close(self) -> None:
            events.append("pool.close")

    pool = FakePool()

    async def fake_create_pool(
        database_url: str, *, desktop_mode: bool = False
    ) -> FakePool:
        assert database_url == "postgresql://resolved"
        assert desktop_mode is True
        return pool

    async def fake_run_migrations(value: object) -> None:
        assert value is pool

    class FakeStore:
        def __init__(self, value: object, **kwargs: object) -> None:
            assert value is pool
            assert callable(kwargs["now"])

        async def fail_stale_generating_answers(self) -> int:
            return 0

    class FakeLlm:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        async def close(self) -> None:
            events.append("llm.close")

    class FakeEmbedding:
        space = EMBEDDING_SPACE

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            events.append("embedding.close")

    class FakeCoordinator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            captured["coordinator_kwargs"] = kwargs

        def has_active_generation(self) -> bool:
            return bool(captured["foreground_active"])

        async def close(self) -> None:
            events.append("coordinator.close")

    class FakeDraftContextLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            key = (
                "proactive_context_kwargs"
                if kwargs.get("system_prompt") == ""
                else "draft_context_kwargs"
            )
            captured[key] = kwargs

    class FakeDraftGeneration:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            captured["draft_generation_kwargs"] = kwargs

    class FakeVoiceDraftCoordinator:
        def __init__(self, **kwargs: object) -> None:
            captured["voice_draft_kwargs"] = kwargs

        async def close(self) -> None:
            events.append("voice_drafts.close")

    class FakeConnector:
        def __init__(self) -> None:
            self.probe_calls = 0

        async def probe(self, target: object) -> None:
            del target
            self.probe_calls += 1
            raise AssertionError("lifespan must not probe Hermes")

        async def close(self) -> None:
            events.append("connector.close")

    connector = FakeConnector()

    class FakeHermesConnectorFactory:
        descriptor = ConnectorDescriptor(
            type_slug="hermes",
            contract_version=1,
            display_name="Hermes Agent",
        )

        def create(self, runtime: object) -> FakeConnector:
            captured["connector_runtime"] = runtime
            return connector

        def target_from_url(self, *args, **kwargs):
            from fish_memory.agent.hermes import HermesConnectorFactory

            return HermesConnectorFactory().target_from_url(*args, **kwargs)

    class FakeAgentCredentialResolver:
        def __init__(self, credentials: object, *, credentials_file: object) -> None:
            captured["credential_resolver_init"] = (credentials, credentials_file)
            self.values = {
                reference: secret.get_secret_value()
                for reference, secret in cast(dict[str, SecretStr], credentials).items()
            }
            self.store_calls: list[tuple[str, str]] = []
            self.global_store_calls: list[tuple[str, str]] = []
            self.clear_global_calls = 0

        def store(self, credential_ref: str, secret: str) -> None:
            self.values[credential_ref] = secret
            self.store_calls.append((credential_ref, secret))

        def resolve(self, credential_ref: str) -> str:
            return self.values[credential_ref]

        def store_global(self, credential_ref: str, secret: str) -> None:
            self.values[credential_ref] = secret
            self.global_store_calls.append((credential_ref, secret))

        def clear_global_credentials(self) -> None:
            self.clear_global_calls += 1
            self.values = {
                reference: secret
                for reference, secret in self.values.items()
                if not reference.startswith("global:")
            }

    async def fake_ensure_global_agent_target(
        store: object,
        factory: object,
        agent_url: str,
    ) -> object:
        captured["global_target_call"] = (store, factory, agent_url)
        return SimpleNamespace(target=SimpleNamespace(id=UUID(int=901)))

    class FakeRunProcessor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["run_processor_args"] = args
            captured["run_processor_kwargs"] = kwargs

        async def process_once(self) -> bool:
            raise AssertionError("lifespan must not synchronously process Agent work")

        async def close(self) -> None:
            events.append("run_processor.close")

    class FakeMessageComposer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["message_composer_args"] = args
            captured["message_composer_kwargs"] = kwargs

    class FakeMessageProcessor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["message_processor_args"] = args
            captured["message_processor_kwargs"] = kwargs
            self.foreground_gate = kwargs["foreground_gate"]

        async def close(self) -> None:
            events.append("message_processor.close")

    class FakeSupervisor:
        def __init__(
            self,
            processor: object,
            *,
            name: str,
            poll_interval_seconds: float,
        ) -> None:
            del poll_interval_seconds
            self.processor = processor
            self.name = name
            cast(list[str], captured.setdefault("supervisor_names", [])).append(name)

        def start(self) -> None:
            events.append(f"supervisor.start:{self.name}")

        async def close(self) -> None:
            events.append(f"supervisor.close:{self.name}")

    monkeypatch.setattr(app_module, "resolve_database", fake_resolve_database)
    monkeypatch.setattr(app_module, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(app_module, "MessagesStore", FakeStore)
    monkeypatch.setattr(app_module, "BailianChatClient", FakeLlm)
    monkeypatch.setattr(
        app_module,
        "create_embedding_client",
        lambda settings: FakeEmbedding(),
    )
    monkeypatch.setattr(
        app_module,
        "ensure_embedding_hnsw_index",
        lambda pool, dimensions: asyncio.sleep(0),
    )
    monkeypatch.setattr(app_module, "AnswerCoordinator", FakeCoordinator)
    monkeypatch.setattr(app_module, "AnswerContextLoader", FakeDraftContextLoader)
    monkeypatch.setattr(
        app_module,
        "ParallelAnswerGenerationPipeline",
        FakeDraftGeneration,
    )
    monkeypatch.setattr(
        app_module,
        "VoiceDraftCoordinator",
        FakeVoiceDraftCoordinator,
    )
    monkeypatch.setattr(
        app_module,
        "HermesConnectorFactory",
        FakeHermesConnectorFactory,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AgentCredentialResolver",
        FakeAgentCredentialResolver,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "global_agent_credential_ref",
        lambda factory, agent_url: "global:test-url",
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "ensure_global_agent_target",
        fake_ensure_global_agent_target,
        raising=False,
    )
    monkeypatch.setattr(
        "fish_memory.agent.configuration.ensure_global_agent_target",
        fake_ensure_global_agent_target,
    )
    monkeypatch.setattr(
        app_module,
        "AgentRunProcessor",
        FakeRunProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AgentMessageComposer",
        FakeMessageComposer,
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "AgentMessageProcessor",
        FakeMessageProcessor,
        raising=False,
    )
    monkeypatch.setattr(app_module, "BackgroundTaskSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        app_module,
        "release_unused_memory",
        lambda reason: None,
    )
    captured["connector"] = connector
    captured["pool"] = pool
    return events, captured


def test_app_registers_agent_message_and_target_routes() -> None:
    app = app_module.create_app(_agent_runtime_settings(enabled=False))
    paths = {
        route.path
        for included in app.routes
        if hasattr(included, "original_router")
        for route in included.original_router.routes
    }
    paths.update(route.path for route in app.routes if hasattr(route, "path"))

    assert "/v1/fishes/{fish_id}/agent-messages/events" in paths
    assert "/v1/fishes/{fish_id}/companion-messages/events" in paths
    assert "/v1/fishes/{fish_id}/companion-messages/{message_id}/consumption" in paths
    assert "/internal/fishes/{fish_id}/observations/proactivity" not in paths
    assert "/internal/fishes/{fish_id}/agent-config" in paths
    assert "/internal/agent-targets" not in paths
    assert "/internal/fishes/{fish_id}/agent-target" not in paths


@pytest.mark.asyncio
async def test_enabled_agent_lifespan_wires_shared_runtime_and_closes_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, captured = _install_agent_runtime_lifespan_fakes(monkeypatch)
    settings = _agent_runtime_settings(enabled=True)
    app = app_module.create_app(settings)

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.agent_target_store, TargetStore)
        assert isinstance(app.state.agent_task_store, TaskStore)
        assert app.state.agent_connector_registry.require("hermes") is not None
        assert app.state.agent_connectors == {
            "hermes": captured["connector"],
        }
        assert app.state.agent_global_configured is True
        assert app.state.agent_global_url == settings.agent_url
        assert app.state.agent_task_route_service is not None
        assert app.state.agent_reply_service is not None
        assert app.state.agent_run_processor is not None
        assert app.state.agent_message_processor is not None
        assert not hasattr(app.state, "proactive_observation_coordinator")

        coordinator_kwargs = cast(dict[str, object], captured["coordinator_kwargs"])
        draft_context_kwargs = cast(dict[str, object], captured["draft_context_kwargs"])
        draft_generation_kwargs = cast(
            dict[str, object], captured["draft_generation_kwargs"]
        )
        assert coordinator_kwargs["agent_target_store"] is app.state.agent_target_store
        assert coordinator_kwargs["agent_task_store"] is app.state.agent_task_store
        assert coordinator_kwargs["agent_default_target_id"] == UUID(int=901)
        assert (
            coordinator_kwargs["agent_task_route_service"]
            is app.state.agent_task_route_service
        )
        assert (
            draft_context_kwargs["agent_target_store"] is app.state.agent_target_store
        )
        assert draft_context_kwargs["agent_task_store"] is app.state.agent_task_store
        assert draft_context_kwargs["agent_default_target_id"] == UUID(int=901)
        assert (
            draft_generation_kwargs["task_route_service"]
            is app.state.agent_task_route_service
        )
        assert (
            coordinator_kwargs["agent_capability_policy"]
            is draft_context_kwargs["agent_capability_policy"]
        )
        assert (
            coordinator_kwargs["environment_context_loader"]
            is draft_context_kwargs["environment_context_loader"]
        )
        assert isinstance(
            coordinator_kwargs["environment_context_loader"],
            app_module.EnvironmentAnswerContextLoader,
        )
        assert callable(coordinator_kwargs["on_ask_submitted"])
        capability_policy = coordinator_kwargs["agent_capability_policy"]
        assert cast(Any, capability_policy).supported_connector_types == frozenset(
            {"hermes"}
        )

        connector_runtime = cast(AgentConnectorRuntime, captured["connector_runtime"])
        assert connector_runtime.request_timeout_seconds == 7.0
        assert connector_runtime.sse_read_timeout_seconds == 11.0
        assert connector_runtime.credential_resolver.resolve("global:test-url") == (
            "provider-key"
        )
        assert connector_runtime.credential_resolver.store_calls == []
        assert connector_runtime.credential_resolver.global_store_calls == [
            ("global:test-url", "provider-key")
        ]
        initial_credentials = cast(
            dict[str, SecretStr],
            cast(tuple[object, object], captured["credential_resolver_init"])[0],
        )
        assert initial_credentials == {}
        assert cast(tuple[object, ...], captured["global_target_call"])[2] == (
            settings.agent_url
        )
        run_processor_args = cast(tuple[object, ...], captured["run_processor_args"])
        run_processor_kwargs = cast(dict[str, object], captured["run_processor_kwargs"])
        assert run_processor_args == (
            app.state.agent_task_store,
            app.state.agent_connectors,
        )
        assert run_processor_kwargs["concurrency"] == 3
        assert run_processor_kwargs["poll_interval_seconds"] == 0.125
        assert run_processor_kwargs["stale_timeout_seconds"] == 19.0
        session_context_loader = run_processor_kwargs["session_context_loader"]
        assert isinstance(session_context_loader, app_module.AgentSessionContextLoader)
        message_processor_kwargs = cast(
            dict[str, object], captured["message_processor_kwargs"]
        )
        background_semaphore = cast(
            asyncio.Semaphore,
            message_processor_kwargs["background_semaphore"],
        )
        assert background_semaphore._value == 2

        names = cast(list[str], captured["supervisor_names"])
        assert "agent-run-processor" in names
        assert "agent-message-processor" in names
        message_processor = cast(
            Any,
            app.state.agent_message_processor,
        )
        gate = message_processor.foreground_gate
        assert await gate.is_active() is False
        captured["foreground_active"] = True
        assert await gate.is_active() is True
        assert cast(Any, captured["connector"]).probe_calls == 0

    supervisor_close_indexes = [
        index
        for index, event in enumerate(events)
        if event.startswith("supervisor.close:")
    ]
    assert supervisor_close_indexes
    processor_close_indexes = [
        events.index("run_processor.close"),
        events.index("message_processor.close"),
    ]
    assert max(supervisor_close_indexes) < min(processor_close_indexes)
    assert max(processor_close_indexes) < events.index("connector.close")
    assert events.index("connector.close") < events.index("coordinator.close")
    assert events.index("coordinator.close") < events.index("pool.close")
    assert events.index("pool.close") < events.index("database.close")


@pytest.mark.asyncio
async def test_task_toggle_keeps_running_host_and_plugin_manager(monkeypatch):
    from fish_memory.runtime_configuration import RuntimeConfiguration

    _events, captured = _install_agent_runtime_lifespan_fakes(monkeypatch)
    settings = _agent_runtime_settings(enabled=False)
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        original = (
            app.state.pool,
            app.state.plugin_manager,
            app.state.extension_service,
            app.state.answer_coordinator,
            app.state.voice_draft_coordinator,
            app.state.agent_run_processor,
        )
        runtime = app.state.agent_runtime_state
        assert not runtime.enabled
        body = RuntimeConfiguration.model_validate(
            {
                "models": settings.resolved_llm_routes(),
                "speech": {"provider": "local"},
                "voice_draft_enabled": True,
                "agent": {
                    "enabled": True,
                    "url": settings.agent_url,
                    "api_key": "provider-key",
                },
            }
        )
        await app.state.runtime_configuration.apply(body)
        assert runtime.enabled and app.state.extension_service.tasks.available
        assert app.state.agent_default_target_id == runtime.default_target_id
        assert app.state.agent_configuration_writable
        revision = runtime.revision
        await app.state.runtime_configuration.apply(body)
        assert runtime.revision == revision  # An unchanged Save is a no-op for Tasks.
        await app.state.runtime_configuration.apply(
            body.model_copy(
                update={
                    "agent": body.agent.model_copy(update={"enabled": False}),
                }
            )
        )
        assert not runtime.enabled and not app.state.extension_service.tasks.available
        assert original == (
            app.state.pool,
            app.state.plugin_manager,
            app.state.extension_service,
            app.state.answer_coordinator,
            app.state.voice_draft_coordinator,
            app.state.agent_run_processor,
        )
        assert captured["connector"].probe_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_primary_without_global_key_revokes_persisted_global_keys(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    _events, captured = _install_agent_runtime_lifespan_fakes(monkeypatch)
    settings = _agent_runtime_settings(enabled=enabled)
    settings.agent_api_key = None
    app = app_module.create_app(settings)

    async with app.router.lifespan_context(app):
        connector_runtime = cast(
            AgentConnectorRuntime,
            captured["connector_runtime"],
        )
        assert app.state.agent_global_configured is False
        assert connector_runtime.credential_resolver.clear_global_calls == 1
        assert connector_runtime.credential_resolver.global_store_calls == []
        assert "global_target_call" not in captured


@pytest.mark.asyncio
async def test_disabled_lifespan_keeps_agent_configuration_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _events, captured = _install_agent_runtime_lifespan_fakes(monkeypatch)
    app = app_module.create_app(_agent_runtime_settings(enabled=False))

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.agent_target_store, TargetStore)
        assert isinstance(app.state.agent_task_store, TaskStore)
        assert app.state.agent_connector_registry.require("hermes") is not None
        assert app.state.agent_connectors == {
            "hermes": captured["connector"],
        }
        # Desktop Task owners stay assembled but dormant until enabled.
        assert not app.state.agent_runtime_state.enabled
        assert not app.state.agent_configuration_writable
        assert not app.state.extension_service.tasks.available
        for key in (
            "coordinator_kwargs",
            "draft_context_kwargs",
            "proactive_context_kwargs",
        ):
            assert captured[key]["agent_runtime_state"] is app.state.agent_runtime_state
        assert cast(Any, captured["connector"]).probe_calls == 0
