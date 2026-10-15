import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fish_memory import failures
from fish_memory.agent.autonomous import AutonomousRunPolicy
from fish_memory.agent.configuration import (
    agent_on_this_computer,
    configure_global_agent,
    ensure_global_agent_target,
    global_agent_credential_ref,
)
from fish_memory.agent.connector import (
    AgentConnector,
    AgentConnectorRegistry,
    AgentConnectorRuntime,
)
from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.hermes import HermesConnectorFactory
from fish_memory.agent.message_delivery import (
    AgentMessageComposer,
    AgentMessageProcessor,
)
from fish_memory.agent.processor import AgentRunProcessor
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyService,
    LlmReplyClassifier,
)
from fish_memory.agent.routes import agent_router, agent_target_router
from fish_memory.agent.routing import (
    AgentTaskRouteService,
    LocalConnectorCapabilityPolicy,
)
from fish_memory.agent.runtime_state import AgentRuntimeState, EnabledAgentProcessor
from fish_memory.agent.session_context import AgentSessionContextLoader
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.environment_context import (
    Abilities,
    EnvironmentAnswerContextLoader,
)
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.parallel_generation import (
    ParallelAnswerGenerationPipeline,
)
from fish_memory.answer.prepared_answer import PreparedAnswer
from fish_memory.answer.prepared_reply import PreparedReplyTurn
from fish_memory.answer.rquery import RQueryIntegrityService
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.answer.voice_drafts import (
    VoiceDraftCoordinator,
)
from fish_memory.api.routes import (
    catalog_router,
    internal_router,
    router,
)
from fish_memory.auth.credentials import credential_store_from_settings
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.routes import auth_router
from fish_memory.auth.service import ApiAuthService
from fish_memory.background.fact_processor import FactProcessor
from fish_memory.background.mask_processor import MaskEmbeddingProcessor
from fish_memory.background.relation_processor import SnapshotItemRelationProcessor
from fish_memory.background.snapshot_processor import DynamicSnapshotProcessor
from fish_memory.background.static_profile_processor import StaticProfileProcessor
from fish_memory.background.supervisor import BackgroundTaskSupervisor
from fish_memory.background_messages.models import BackgroundMessageSource
from fish_memory.background_messages.processor import BackgroundMessageExpiryProcessor
from fish_memory.background_messages.routes import background_message_router
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.clock import Clock, SystemClock
from fish_memory.codex_notifications.client import CodexReadClient
from fish_memory.codex_notifications.delivery import CompletionDelivery
from fish_memory.codex_notifications.observer import CompletionObserver
from fish_memory.config import Settings
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import TaskStore
from fish_memory.db.autonomous_tasks import AutonomousTaskStore
from fish_memory.db.embedded import resolve_database
from fish_memory.db.embedding_spaces import (
    FishEmbeddingSpaceStore,
    ensure_embedding_hnsw_index,
)
from fish_memory.db.entities import EntityStore
from fish_memory.db.facts import FactStore
from fish_memory.db.fishes import FishStore
from fish_memory.db.home_actions import HomeActionStore
from fish_memory.db.memory_forget import MemoryForgetStore
from fish_memory.db.message_attachments import MessageAttachmentStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.performance import PerformanceStore
from fish_memory.db.plugin_tasks import PluginTaskStore
from fish_memory.db.pool import create_pool
from fish_memory.db.recall import RecallStore
from fish_memory.db.relations import SnapshotItemRelationStore
from fish_memory.db.sessions import SessionStore
from fish_memory.db.snapshots import SnapshotStore
from fish_memory.embeddings.client import EmbeddingClient
from fish_memory.embeddings.factory import create_embedding_client
from fish_memory.entities.resolver import EntityResolver
from fish_memory.environment_frame_routes import environment_frame_router
from fish_memory.environment_frames import EnvironmentFrameCache
from fish_memory.experimental.jev import JevClient
from fish_memory.experimental.judge import CompanionJudge, ReasoningJudge
from fish_memory.experimental.materials import MaterialStore
from fish_memory.experimental.perception import ExperimentalPerception
from fish_memory.experimental.runtime import ExperimentalRuntime
from fish_memory.experimental.service import ExperimentalService
from fish_memory.extensions.context import ExtensionContext
from fish_memory.extensions.manager import PluginManager
from fish_memory.extensions.reviews import PluginReviews
from fish_memory.extensions.routes import extension_router
from fish_memory.extensions.service import ExtensionService
from fish_memory.extensions.store import ExtensionStore
from fish_memory.extensions.tasks import PluginTasks
from fish_memory.facts.extractor import FactExtractor
from fish_memory.gateway.delivery import GatewayDelivery
from fish_memory.gateway.routes import router as gateway_router
from fish_memory.history.routes import history_router
from fish_memory.history.service import SessionHistoryService
from fish_memory.history.store import SessionHistoryStore
from fish_memory.home.client import HomeAssistantClient
from fish_memory.home.detection import looks_like_home_request
from fish_memory.home.router import HomeRouter
from fish_memory.home.routes import home_router
from fish_memory.home.scope_planner import HomeScopePlanner
from fish_memory.home.selector import HomeDeviceSelector
from fish_memory.home.service import HomeAutomationService
from fish_memory.kws.routes import kws_router
from fish_memory.llm.client import BailianChatClient
from fish_memory.llm.log_routes import router as llm_log_router
from fish_memory.llm.log_store import LOG_CLEANUP_INTERVAL_SECONDS, LlmLogStore
from fish_memory.llm.logs import CallJournal
from fish_memory.llm.runtime import ModelRuntime
from fish_memory.observations.computer_history import ComputerHistorySummaryService
from fish_memory.observations.environment_history_audio import (
    LocalSystemAudioTranscriber,
)
from fish_memory.observations.environment_history_client import (
    EnvironmentHistoryModelClient,
)
from fish_memory.observations.environment_history_routes import (
    environment_history_router,
)
from fish_memory.observations.environment_history_service import (
    CombinedEnvironmentSummaryLoader,
    EnvironmentHistoryService,
)
from fish_memory.observations.environment_history_store import EnvironmentHistoryStore
from fish_memory.observations.routes import (
    observation_internal_router,
    observation_router,
)
from fish_memory.observations.service import ObservationService
from fish_memory.observations.store import ObservationStore
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.routes import router as pet_router
from fish_memory.proactive.coordinator import ProactiveCoordinator
from fish_memory.proactive.delivery import ProactiveDelivery
from fish_memory.proactive.expression import ExpressionComposer, ExpressionRegistry
from fish_memory.proactive.routes import proactive_router
from fish_memory.proactive.runtime import ProactiveRuntimeBridge
from fish_memory.proactive.tasks import AutonomousTaskExecutor
from fish_memory.recall.planner import RecallPlanner
from fish_memory.recall.service import FactRecallService
from fish_memory.runtime_configuration import (
    AgentConfiguration,
    AgentTuning,
    RuntimeConfigurationService,
    runtime_configuration_router,
)
from fish_memory.runtime_memory import release_unused_memory
from fish_memory.snapshots.confirmer import (
    SnapshotUpdateConfirmer,
    StaticProfileUpdateConfirmer,
)
from fish_memory.snapshots.static_updater import StaticProfileUpdater
from fish_memory.snapshots.updater import DynamicSnapshotUpdater
from fish_memory.tts.custom_voices import DashScopeVoiceClient
from fish_memory.tts.local import prefetch_local_tts_model, voice_provider
from fish_memory.tts.routes import tts_internal_router, tts_router
from fish_memory.tts.runtime import CustomVoiceRuntime
from fish_memory.tts.speech_tags import supports_speech_tags
from fish_memory.tts.voice_config import resolve_voice_config

logger = logging.getLogger(__name__)


class _AgentMessageForegroundGate:
    def __init__(self, coordinator: AnswerCoordinator) -> None:
        self._coordinator = coordinator

    async def is_active(self) -> bool:
        return self._coordinator.has_active_generation()


def _voice_draft_preparer(
    context_loader: AnswerContextLoader,
    generation: ParallelAnswerGenerationPipeline,
    *,
    enable_home: bool,
    reply_service: AgentReplyService | None = None,
) -> Callable[..., Awaitable[PreparedAnswer]]:
    async def prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
        *,
        created_at: datetime,
        reply_context: AgentReplyContext | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> PreparedAnswer:
        tracker = MessagePerformanceTracker()
        prepared_reply = None
        if reply_context is not None:
            if reply_service is None:
                raise AgentReplyConflictError("Agent reply service is unavailable")
            prepared_reply = await reply_service.prepare_reply(
                key.fish_id,
                key.session_id,
                content,
                reply_context,
                message_id=key.turn_id,
            )
            if not prepared_reply.requires_answer:
                return PreparedReplyTurn(prepared_reply)
        loaded = await context_loader.load_for_draft(
            key.fish_id,
            key.session_id,
            turn_id=key.turn_id,
            content=content,
            created_at=created_at,
            tracker=tracker,
            **(
                {
                    "environment_revision": revision,
                    "speech_started_at": speech_started_at,
                    "frame_snapshot_at": frame_snapshot_at,
                }
                if speech_started_at is not None and frame_snapshot_at is not None
                else {}
            ),
        )
        return generation.prepare_initial(
            key.fish_id,
            loaded.messages,
            loaded.recall_conversation,
            PromotableAnswerSink(),
            tracker,
            enable_home=enable_home and looks_like_home_request(content),
            agent_target=loaded.agent_target,
            task_lane_context=loaded.task_lane_context,
            prepared_reply=prepared_reply,
            **(
                {"activity_context": loaded.activity_context}
                if loaded.activity_context
                else {}
            ),
        )

    return prepare


async def image_input_capability(model: Any) -> str:
    """A model's image input, sending the small probe image when unknown."""
    if model is None:
        return "unknown"
    capability = getattr(model, "image_input_capability", "unknown")
    probe = getattr(model, "probe_image_input_capability", None)
    if capability != "unknown" or not callable(probe):
        return str(capability)
    try:
        return str(await asyncio.wait_for(probe(), timeout=30))
    except Exception:
        return "unknown"


def create_app(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
) -> FastAPI:
    app_clock = clock or SystemClock()
    resolved_settings = settings or Settings()
    credentials = credential_store_from_settings(resolved_settings)
    auth_service = ApiAuthService(credentials)
    pet_event_hub = PetEventHub()
    pet_event_hub.publish_control(
        "pet.microphone",
        None,
        {"enabled": resolved_settings.companion_microphone_enabled},
    )
    pet_event_hub.publish_control(
        "pet.speaker", None, {"enabled": resolved_settings.companion_playback_enabled}
    )
    environment_frame_cache = EnvironmentFrameCache()
    agent_runtime_owned = (
        resolved_settings.desktop_mode or resolved_settings.agent_tasks_enabled
    )

    def publish_service_problems() -> None:
        # A missing key or spent quota stays on the desktop until that service
        # works again; the hub replays the latest list to a new desktop.
        pet_event_hub.publish_control(
            "pet.service.problems",
            None,
            {"problems": [problem.code for problem in failures.health.problems()]},
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pet_event_hub.reopen()
        publish_service_problems()
        forget_service_problems = failures.health.subscribe(publish_service_problems)
        database = await resolve_database(
            resolved_settings.database_url,
            data_dir=resolved_settings.database_data_dir,
            desktop_mode=resolved_settings.desktop_mode,
        )
        pool = None
        llm: BailianChatClient | None = None
        reasoning_llm: BailianChatClient | None = None
        model_runtime: ModelRuntime | None = None
        call_journal: CallJournal | None = None
        experiment: ExperimentalService | None = None
        image_capability_probe_task: asyncio.Task[Any] | None = None
        local_tts_prefetch: asyncio.Task[None] | None = None

        def schedule_local_tts_prefetch(force: bool = False) -> None:
            nonlocal local_tts_prefetch
            if local_tts_prefetch is not None and not local_tts_prefetch.done():
                return
            local_tts_prefetch = asyncio.get_running_loop().create_task(
                prefetch_local_tts_model(resolved_settings, force=force),
                name="local-tts-prefetch",
            )

        app.state.prepare_local_voice = lambda: schedule_local_tts_prefetch(True)
        embedder: EmbeddingClient | None = None
        coordinator: AnswerCoordinator | None = None
        voice_draft_coordinator: VoiceDraftCoordinator | None = None
        environment_history_service: EnvironmentHistoryService | None = None
        session_history: SessionHistoryService | None = None
        kws_service: Any | None = None
        custom_voice_client: CustomVoiceRuntime | None = None
        home_client: HomeAssistantClient | None = None
        home_service: HomeAutomationService | None = None
        agent_connectors: dict[str, AgentConnector] = {}
        agent_run_processor: AgentRunProcessor | None = None
        agent_run_supervisor: BackgroundTaskSupervisor | None = None
        agent_message_processor: AgentMessageProcessor | None = None
        proactive_coordinator: ProactiveCoordinator | None = None
        proactive_bridge: ProactiveRuntimeBridge | None = None
        codex_observer: CompletionObserver | None = None
        extension_service: ExtensionService | None = None
        plugin_manager: PluginManager | None = None
        supervisors: list[BackgroundTaskSupervisor] = []
        try:
            if credentials.was_generated:
                logger.info(
                    "Generated fish-memory API key (shown once): %s; "
                    "open /unlock on this server",
                    credentials.current_key(),
                )
            else:
                logger.info("fish-memory API authentication is enabled")
            logger.info("Initializing database connection pool")
            pool = await create_pool(
                database.database_url,
                desktop_mode=resolved_settings.desktop_mode,
            )
            logger.info("Applying database migrations")
            await run_migrations(cast(MigrationPool, pool))
            logger.info(
                "Initializing %s embeddings",
                resolved_settings.embedding_provider,
            )
            embedder = create_embedding_client(resolved_settings)
            await embedder.initialize()
            logger.info(
                "Ensuring %d-dimensional embedding index",
                embedder.space.dimensions,
            )
            await ensure_embedding_hnsw_index(pool, embedder.space.dimensions)
            embedding_space_store = FishEmbeddingSpaceStore(pool)
            if resolved_settings.desktop_mode:
                logger.info(
                    "Desktop Companion owns the KWS session; API duplicate disabled"
                )
            else:
                from fish_memory.kws.service import KwsService

                kws_service = KwsService.from_settings(resolved_settings)
                await kws_service.start()
            message_attachments = MessageAttachmentStore(pool)
            store = MessagesStore(
                pool,
                embedding_space=embedder.space,
                now=app_clock.now,
                history_token_limit=resolved_settings.history_token_limit,
                history_token_retain=resolved_settings.history_token_retain,
            )
            fish_store = FishStore(pool, embedding_space=embedder.space)
            session_store = SessionStore(pool, now=app_clock.now)
            background_message_store = BackgroundMessageStore(
                pool,
                messages_store=store,
                now=app_clock.now,
            )
            agent_target_store = TargetStore(pool)
            agent_task_store = TaskStore(
                pool,
                messages_store=store,
                background_message_store=background_message_store,
            )
            agent_connector_registry = AgentConnectorRegistry()
            hermes_factory = HermesConnectorFactory()
            agent_connector_registry.register(hermes_factory)
            agent_global_url = resolved_settings.agent_url
            agent_global_configured = bool(
                agent_runtime_owned
                and resolved_settings.agent_url is not None
                and resolved_settings.agent_api_key is not None
            )
            agent_global_credential_ref = (
                global_agent_credential_ref(
                    hermes_factory,
                    resolved_settings.agent_url,
                )
                if agent_global_configured and resolved_settings.agent_url is not None
                else None
            )
            agent_credential_resolver = AgentCredentialResolver(
                {},
                credentials_file=(
                    Path(resolved_settings.agent_credentials_file)
                    if agent_runtime_owned or resolved_settings.agent_api_key is None
                    else None
                ),
            )
            if (
                agent_global_configured
                and agent_global_credential_ref is not None
                and resolved_settings.agent_api_key is not None
            ):
                agent_credential_resolver.store_global(
                    agent_global_credential_ref,
                    resolved_settings.agent_api_key.get_secret_value(),
                )
            elif resolved_settings.agent_api_key is None:
                agent_credential_resolver.clear_global_credentials()
            hermes_connector = hermes_factory.create(
                AgentConnectorRuntime(
                    credential_resolver=agent_credential_resolver,
                    request_timeout_seconds=(
                        resolved_settings.agent_http_timeout_seconds
                    ),
                    sse_read_timeout_seconds=(
                        resolved_settings.agent_sse_read_timeout_seconds
                    ),
                )
            )
            agent_connectors = {
                hermes_factory.descriptor.type_slug: hermes_connector,
            }
            agent_default_target_id: UUID | None = None
            if agent_global_configured and resolved_settings.agent_url is not None:
                global_agent_target = await ensure_global_agent_target(
                    agent_target_store,
                    hermes_factory,
                    resolved_settings.agent_url,
                )
                agent_default_target_id = global_agent_target.target.id
            agent_runtime_state = AgentRuntimeState(
                enabled=agent_runtime_owned and resolved_settings.agent_tasks_enabled,
                default_target_id=agent_default_target_id,
            )
            await store.fail_stale_generating_answers()
            performance_store = PerformanceStore(pool)
            llm_log_store = LlmLogStore(
                pool,
                retention_days=lambda: resolved_settings.llm_log_retention_days,
            )
            if resolved_settings.desktop_mode:
                await llm_log_store.recover()
            call_journal = CallJournal(llm_log_store)
            app.state.llm_log_store = llm_log_store
            app.state.llm_call_journal = call_journal
            home_action_store = HomeActionStore(pool)
            observation_store = ObservationStore(pool)
            observation_service = ObservationService(
                observation_store,
                clock=app_clock,
            )
            model_routes = resolved_settings.resolved_llm_routes()
            model_runtime = ModelRuntime(
                model_routes,
                text_factory=lambda **kwargs: BailianChatClient(
                    **kwargs, call_journal=call_journal
                ),
                environment_factory=lambda **kwargs: EnvironmentHistoryModelClient(
                    **kwargs, call_journal=call_journal
                ),
            )
            llm = cast(BailianChatClient, model_runtime.chat)
            reasoning_llm = cast(BailianChatClient, model_runtime.reasoning)

            def notify_history_boundary(
                fish: UUID, session: UUID, at: datetime
            ) -> None:
                pet_event_hub.publish_control(
                    "history.boundary",
                    None,
                    {
                        "fishId": str(fish),
                        "sessionId": str(session),
                        "occurredAt": at.timestamp(),
                    },
                )

            session_history = SessionHistoryService(
                SessionHistoryStore(pool),
                reasoning_llm,
                now=app_clock.now,
                token_limit=resolved_settings.history_token_limit,
                on_boundary=notify_history_boundary,
            )
            session_history.start()
            app.state.session_history = session_history
            computer_history_summary_service = ComputerHistorySummaryService(
                observation_store,
                reasoning_llm,
            )
            environment_history_store = EnvironmentHistoryStore(pool)
            environment_history_service = EnvironmentHistoryService(
                environment_history_store,
                cast(EnvironmentHistoryModelClient, model_runtime.environment),
                LocalSystemAudioTranscriber(resolved_settings),
                now=app_clock.now,
                # Screen summaries run on the reasoning model's service.
                image_input=lambda: image_input_capability(reasoning_llm),
            )
            environment_context_loader = EnvironmentAnswerContextLoader(
                CombinedEnvironmentSummaryLoader(
                    environment_history_service,
                    computer_history_summary_service,
                    policy=environment_frame_cache.policy,
                ),
                environment_frame_cache,
                llm,
                now=app_clock.now,
                # Read on every turn: the microphone and Agent switch live.
                abilities=lambda: Abilities(
                    microphone=resolved_settings.companion_microphone_enabled,
                    search=resolved_settings.answer_enable_search,
                    agent=resolved_settings.agent_tasks_enabled,
                    agent_local=agent_on_this_computer(resolved_settings.agent_url),
                    home=resolved_settings.home_assistant_enabled,
                ),
            )

            def notify_computer_history_refresh(
                fish_id: UUID,
                session_id: UUID,
            ) -> None:
                session_history.set_active(fish_id, session_id)
                if proactive_coordinator is not None:
                    proactive_coordinator.conversation_started(fish_id, session_id)
                if not environment_frame_cache.policy(fish_id).operation_enabled:
                    return
                pet_event_hub.publish_control(
                    "computer_history.refresh_requested",
                    None,
                    {
                        "fishId": str(fish_id),
                        "sessionId": str(session_id),
                    },
                )

            def request_environment_image_capability_probe() -> None:
                nonlocal image_capability_probe_task
                if llm is None or llm.image_input_capability != "unknown":
                    return
                if (
                    image_capability_probe_task is not None
                    and not image_capability_probe_task.done()
                ):
                    return
                image_capability_probe_task = asyncio.create_task(
                    llm.probe_image_input_capability(),
                    name="environment-image-capability-probe",
                )

            agent_capability_policy = (
                LocalConnectorCapabilityPolicy(
                    supported_connector_types=frozenset(agent_connectors),
                    steer_connector_types=frozenset(
                        connector_type
                        for connector_type in agent_connectors
                        if connector_type == "hermes"
                    ),
                )
                if agent_runtime_owned
                else None
            )
            agent_reply_service = (
                AgentReplyService(agent_task_store, LlmReplyClassifier(reasoning_llm))
                if agent_runtime_owned
                else None
            )
            agent_task_route_service = (
                AgentTaskRouteService(
                    agent_task_store,
                    interaction_reply_service=agent_reply_service,
                    runtime_state=agent_runtime_state,
                )
                if agent_runtime_owned
                else None
            )
            if resolved_settings.home_assistant_enabled:
                token = resolved_settings.home_assistant_token
                assert token is not None
                home_client = HomeAssistantClient(
                    base_url=resolved_settings.home_assistant_url,
                    token=token.get_secret_value(),
                )
                home_service = HomeAutomationService(
                    client=home_client,
                    store=home_action_store,
                    history_store=store,
                    router=HomeRouter(reasoning_llm),
                    scope_planner=HomeScopePlanner(reasoning_llm),
                    selector=HomeDeviceSelector(reasoning_llm),
                    performance_store=performance_store,
                    timeout_seconds=resolved_settings.home_router_timeout_seconds,
                    concurrency=resolved_settings.home_router_concurrency,
                )
                await home_client.start()
            snapshot_store = SnapshotStore(pool)
            entity_store = EntityStore(pool)
            snapshot_max_age = timedelta(
                days=resolved_settings.dynamic_snapshot_max_age_days
            )
            recall_store = RecallStore(pool)
            recall_service = FactRecallService(
                recall_store,
                RecallPlanner(reasoning_llm),
                embedder,
                embedding_space_store,
                now=app_clock.now,
            )
            rquery_integrity_service = RQueryIntegrityService(
                recall_store, reasoning_llm
            )
            experiment = ExperimentalService(
                resolved_settings.experimental_companion,
                judge=CompanionJudge(
                    JevClient(call_journal), ReasoningJudge(reasoning_llm)
                ),
                chat=llm,
                reasoning=reasoning_llm,
                recall=recall_service,
                observations=observation_store,
                now=app_clock.now,
                search_enabled=resolved_settings.answer_enable_search,
                recall_token_budget=resolved_settings.recall_context_token_budget,
                perception=ExperimentalPerception(
                    environment_frame_cache, environment_history_service
                ),
            )
            coordinator_agent_kwargs: dict[str, Any] = {}
            if agent_runtime_owned:
                coordinator_agent_kwargs = {
                    "agent_target_store": agent_target_store,
                    "agent_task_store": agent_task_store,
                    "agent_default_target_id": agent_default_target_id,
                    "agent_capability_policy": agent_capability_policy,
                    "agent_runtime_state": agent_runtime_state,
                    "agent_task_route_service": agent_task_route_service,
                }

            async def speech_tags_for_fish(fish_id: UUID) -> bool:
                saved = await fish_store.get_voice_config(fish_id)
                provider = voice_provider(saved, app.state.settings)
                voice = resolve_voice_config(saved, provider=provider)
                return supports_speech_tags(provider, voice.config.model)

            coordinator = AnswerCoordinator(
                store,
                llm,
                speech_tags_provider=speech_tags_for_fish,
                experiment=experiment,
                message_attachments=message_attachments,
                performance_store=performance_store,
                fish_store=fish_store,
                snapshot_store=snapshot_store,
                recall_service=recall_service,
                rquery_integrity_service=rquery_integrity_service,
                recall_context_token_budget=(
                    resolved_settings.recall_context_token_budget
                ),
                answer_history_message_limit=(
                    resolved_settings.answer_history_message_limit
                ),
                dynamic_snapshot_max_age=snapshot_max_age,
                now=app_clock.now,
                system_prompt=resolved_settings.system_prompt,
                system_prompt_extension=None,
                activity_context_provider=lambda fish, session: (
                    extension_service.current_activity(fish, session)
                    if extension_service is not None
                    else ""
                ),
                answer_enable_search=resolved_settings.answer_enable_search,
                home_action_service=home_service,
                environment_context_loader=environment_context_loader,
                session_history=session_history,
                on_ask_submitted=notify_computer_history_refresh,
                **coordinator_agent_kwargs,
            )
            if (
                resolved_settings.companion_voice_draft_enabled
                or resolved_settings.desktop_mode
            ):
                draft_agent_kwargs: dict[str, Any] = {}
                if agent_runtime_owned:
                    draft_agent_kwargs = {
                        "agent_target_store": agent_target_store,
                        "agent_task_store": agent_task_store,
                        "agent_default_target_id": agent_default_target_id,
                        "agent_capability_policy": agent_capability_policy,
                        "agent_runtime_state": agent_runtime_state,
                    }
                draft_context_loader = AnswerContextLoader(
                    store,
                    speech_tags_provider=speech_tags_for_fish,
                    snapshot_store=snapshot_store,
                    fish_store=fish_store,
                    environment_context_loader=environment_context_loader,
                    session_history=session_history,
                    history_message_limit=(
                        resolved_settings.answer_history_message_limit
                    ),
                    dynamic_snapshot_max_age=snapshot_max_age,
                    now=app_clock.now,
                    system_prompt=resolved_settings.system_prompt,
                    system_prompt_extension=None,
                    activity_context_provider=lambda fish, session: (
                        extension_service.current_activity(fish, session)
                        if extension_service is not None
                        else ""
                    ),
                    **draft_agent_kwargs,
                )
                draft_generation_agent_kwargs: dict[str, Any] = {}
                if agent_runtime_owned:
                    draft_generation_agent_kwargs["task_route_service"] = (
                        agent_task_route_service
                    )
                draft_generation = ParallelAnswerGenerationPipeline(
                    llm,
                    experiment=experiment,
                    recall_service=recall_service,
                    home_service=home_service,
                    rquery_integrity_service=rquery_integrity_service,
                    recall_context_token_budget=(
                        resolved_settings.recall_context_token_budget
                    ),
                    enable_search=resolved_settings.answer_enable_search,
                    **draft_generation_agent_kwargs,
                )
                voice_draft_coordinator = VoiceDraftCoordinator(
                    prepare=_voice_draft_preparer(
                        draft_context_loader,
                        draft_generation,
                        enable_home=home_service is not None,
                        reply_service=agent_reply_service,
                    ),
                    ttl_seconds=(resolved_settings.companion_voice_draft_ttl_ms / 1000),
                    max_concurrency=(
                        resolved_settings.companion_voice_draft_max_concurrency
                    ),
                    now=app_clock.now,
                )
            fact_store = FactStore(pool)
            fact_llm_semaphore = asyncio.Semaphore(
                resolved_settings.background_llm_concurrency
            )
            fact_embedding_semaphore = asyncio.Semaphore(
                resolved_settings.background_embedding_concurrency
            )
            dynamic_snapshot_llm_semaphore = asyncio.Semaphore(
                resolved_settings.background_llm_concurrency
            )
            static_profile_llm_semaphore = asyncio.Semaphore(
                resolved_settings.background_llm_concurrency
            )
            fact_extractor = FactExtractor(reasoning_llm)
            fact_processor = FactProcessor(
                fact_store,
                fact_extractor,
                embedder,
                performance_store,
                embedding_space_store=embedding_space_store,
                llm_semaphore=fact_llm_semaphore,
                embedding_semaphore=fact_embedding_semaphore,
                max_retries=resolved_settings.fact_extraction_max_retries,
                batch_size=resolved_settings.fact_batch_size,
                llm_timeout_seconds=resolved_settings.fact_llm_timeout_seconds,
                now=app_clock.now,
            )
            snapshot_updater = DynamicSnapshotUpdater(reasoning_llm)
            entity_resolver = EntityResolver(reasoning_llm)
            snapshot_confirmer = SnapshotUpdateConfirmer(reasoning_llm)
            snapshot_processor = DynamicSnapshotProcessor(
                snapshot_store,
                snapshot_updater,
                snapshot_confirmer,
                entity_resolver,
                entity_store,
                performance_store,
                llm_semaphore=dynamic_snapshot_llm_semaphore,
                max_retries=resolved_settings.fact_extraction_max_retries,
                batch_size=resolved_settings.snapshot_batch_size,
                now=app_clock.now,
            )
            static_profile_updater = StaticProfileUpdater(reasoning_llm)
            static_profile_confirmer = StaticProfileUpdateConfirmer(reasoning_llm)
            static_profile_processor = StaticProfileProcessor(
                snapshot_store,
                static_profile_updater,
                static_profile_confirmer,
                entity_resolver,
                entity_store,
                performance_store,
                llm_semaphore=static_profile_llm_semaphore,
                max_retries=resolved_settings.fact_extraction_max_retries,
                batch_size=resolved_settings.snapshot_batch_size,
                now=app_clock.now,
            )
            dynamic_relation_store = SnapshotItemRelationStore(
                pool,
                snapshot_type="dynamic",
            )
            dynamic_relation_processor = SnapshotItemRelationProcessor(
                dynamic_relation_store,
                embedder,
                embedding_space_store=embedding_space_store,
                embedding_semaphore=fact_embedding_semaphore,
                snapshot_type="dynamic",
                max_retries=resolved_settings.fact_extraction_max_retries,
                batch_size=resolved_settings.snapshot_batch_size,
            )
            static_relation_store = SnapshotItemRelationStore(
                pool,
                snapshot_type="static",
            )
            static_relation_processor = SnapshotItemRelationProcessor(
                static_relation_store,
                embedder,
                embedding_space_store=embedding_space_store,
                embedding_semaphore=fact_embedding_semaphore,
                snapshot_type="static",
                max_retries=resolved_settings.fact_extraction_max_retries,
                batch_size=resolved_settings.snapshot_batch_size,
            )
            mask_embedding_processor = MaskEmbeddingProcessor(
                recall_store,
                embedder,
                batch_size=resolved_settings.snapshot_batch_size,
                embedding_semaphore=fact_embedding_semaphore,
            )
            autonomous_policy = None
            plugin_task_policy = None
            plugin_task_store = None
            if agent_runtime_owned:
                autonomous_policy = AutonomousRunPolicy(
                    AutonomousTaskStore(pool),
                    agent_task_store,
                    agent_connectors,
                    timeout_seconds=resolved_settings.proactive_task_timeout_seconds,
                )
                await autonomous_policy.recover()
                plugin_task_store = PluginTaskStore(pool)
                plugin_task_policy = AutonomousRunPolicy(
                    plugin_task_store,
                    agent_task_store,
                    agent_connectors,
                    timeout_seconds=resolved_settings.proactive_task_timeout_seconds,
                    origin="plugin",
                )
                await plugin_task_policy.recover()
                agent_run_processor = AgentRunProcessor(
                    agent_task_store,
                    agent_connectors,
                    concurrency=resolved_settings.agent_run_concurrency,
                    poll_interval_seconds=(
                        resolved_settings.agent_run_poll_interval_seconds
                    ),
                    stale_timeout_seconds=(
                        resolved_settings.agent_run_stale_timeout_seconds
                    ),
                    session_context_loader=AgentSessionContextLoader(store),
                    autonomous_policy=autonomous_policy,
                    plugin_policy=plugin_task_policy,
                    runtime_state=agent_runtime_state,
                )
                agent_message_processor = AgentMessageProcessor(
                    agent_task_store,
                    AgentMessageComposer(llm, fish_store),
                    background_semaphore=asyncio.Semaphore(
                        resolved_settings.background_llm_concurrency
                    ),
                    foreground_gate=_AgentMessageForegroundGate(coordinator),
                )
            proactive_streams = ExpressionRegistry()
            await background_message_store.cancel_orphaned_proactive_streams()

            def cancel_proactive_delivery(fish_id: UUID, message_id: UUID) -> None:
                pet_event_hub.publish_control(
                    "pet.answer.interrupt",
                    None,
                    {"fishId": str(fish_id), "messageId": str(message_id)},
                )

            proactive_expression = ExpressionComposer(
                ProactiveDelivery(background_message_store, proactive_streams)
            )
            proactive_coordinator = ProactiveCoordinator(
                streams=proactive_streams,
                policy=environment_frame_cache.policy,
                tasks=AutonomousTaskExecutor(
                    autonomous_policy,
                    agent_connectors,
                    cooldown_seconds=resolved_settings.proactive_task_cooldown_seconds,
                    starts_per_hour=resolved_settings.proactive_task_starts_per_hour,
                    runtime_state=agent_runtime_state,
                )
                if autonomous_policy is not None
                else None,
            )
            environment_history_service.set_proactivity(proactive_coordinator)
            experiment.tasks = proactive_coordinator.tasks
            experimental_contexts = AnswerContextLoader(
                store,
                speech_tags_provider=speech_tags_for_fish,
                snapshot_store=snapshot_store,
                fish_store=fish_store,
                environment_context_loader=environment_context_loader,
                session_history=session_history,
                history_message_limit=resolved_settings.answer_history_message_limit,
                dynamic_snapshot_max_age=snapshot_max_age,
                now=app_clock.now,
                system_prompt="",
                agent_target_store=agent_target_store if agent_runtime_owned else None,
                agent_task_store=agent_task_store if agent_runtime_owned else None,
                agent_default_target_id=agent_default_target_id,
                agent_capability_policy=agent_capability_policy,
                agent_runtime_state=agent_runtime_state,
            )

            experimental_runtime = ExperimentalRuntime(
                experiment,
                contexts=experimental_contexts,
                expression=proactive_expression,
                streams=proactive_streams,
                foreground_generation=lambda: (
                    (coordinator is not None and coordinator.has_active_generation())
                    or (
                        extension_service is not None
                        and extension_service.has_active_response()
                    )
                ),
                cancel_delivery=cancel_proactive_delivery,
                history=session_history,
                thought_context_state=lambda fish, session: (
                    environment_frame_cache.policy_revision,
                    model_runtime.revision if model_runtime is not None else None,
                    experimental_contexts.current_activity(fish, session),
                ),
                material_store=MaterialStore(pool),
                recall_store=recall_store,
                embedder=embedder,
            )
            proactive_coordinator.experimental = experimental_runtime
            experiment.on_configure = (
                proactive_coordinator.experimental_configuration_changed
            )
            session_history.store.on_unit = experimental_runtime.observe
            app.state.experimental_companion = experiment
            proactive_bridge = ProactiveRuntimeBridge(
                proactive_coordinator, pet_event_hub
            )
            app.state.proactive_coordinator = proactive_coordinator
            extension_store = ExtensionStore(pool)

            async def plugin_session_target() -> tuple[UUID, UUID] | None:
                if experimental_runtime.binding is not None:
                    return experimental_runtime.binding
                fish = await fish_store.get_assistant()
                if fish is None:
                    return None
                session = await session_store.get_latest(fish)
                if session is None:
                    session = await session_store.create(fish)
                return fish, session.id

            extension_service = ExtensionService(
                session_target=plugin_session_target,
                reasoning_llm=reasoning_llm,
                store=extension_store,
                delivery=background_message_store,
                contexts=ExtensionContext(
                    fish_store,
                    store,
                    snapshot_store,
                    environment_context_loader,
                    extension_store,
                    history_limit=resolved_settings.answer_history_message_limit,
                    snapshot_max_age=snapshot_max_age,
                    audiovisual=environment_history_service,
                    session_history=session_history,
                ),
                llm=llm,
                hub=pet_event_hub,
                proactive=proactive_coordinator,
                performance=performance_store,
                on_response_state_changed=experimental_runtime.refresh_busy,
                foreground=lambda: (
                    coordinator is not None and coordinator.has_active_generation()
                ),
            )
            if plugin_task_policy is not None and plugin_task_store is not None:
                extension_service.tasks = PluginTasks(
                    plugin_task_store,
                    agent_target_store,
                    plugin_task_policy,
                    default_target_id=agent_default_target_id,
                    instances=extension_service.instances,
                    emit=extension_service.emit,
                    runtime_state=agent_runtime_state,
                )
                extension_service.tasks.start()
            app.state.extension_service = extension_service

            async def notification_target() -> tuple[UUID, UUID] | None:
                fish_id = await fish_store.get_assistant()
                if fish_id is None:
                    return None
                session = await session_store.get_latest(fish_id)
                if session is None:
                    session = await session_store.create(fish_id)
                return fish_id, session.id

            app.state.gateway_delivery = GatewayDelivery(
                extension_service, notification_target
            )
            if resolved_settings.desktop_mode:
                notice_delivery = CompletionDelivery(
                    extension_service,
                    target=notification_target,
                    enabled=lambda: resolved_settings.codex_notifications_enabled,
                )
                codex_observer = CompletionObserver(
                    CodexReadClient(),
                    Path(resolved_settings.codex_notifications_state_file),
                    notice_delivery,
                    enabled=resolved_settings.codex_notifications_enabled,
                )
                codex_observer.start()
            app.state.codex_completion_observer = codex_observer
            plugin_root = Path(
                os.environ.get("FISH_PLUGIN_ROOT", ".cache/fish-plugins")
            )
            plugin_manager = PluginManager(
                extension_service,
                session_store,
                plugin_root,
                Path(os.environ["FISH_PLUGIN_ASSETS"])
                if os.environ.get("FISH_PLUGIN_ASSETS")
                else None,
                reviews=PluginReviews(
                    root=plugin_root / "reviews",
                    store=extension_store,
                    llm=reasoning_llm,
                    fishes=fish_store,
                    foreground=lambda: (
                        (
                            extension_service is not None
                            and bool(extension_service._foreground)
                        )
                        or (
                            coordinator is not None
                            and coordinator.has_active_generation()
                        )
                    ),
                ),
            )
            app.state.plugin_manager = plugin_manager
            await extension_store.recover()
            await background_message_store.cancel_orphaned_proactive_streams(
                source=BackgroundMessageSource.PLUGIN
            )
            custom_voice_client = CustomVoiceRuntime(
                resolved_settings, factory=DashScopeVoiceClient
            )
            app.state.custom_voice_client = custom_voice_client

            async def configure_agent(configuration: AgentConfiguration) -> None:
                if not agent_runtime_owned:
                    raise RuntimeError("Task runtime is owned by another service")
                if (
                    resolved_settings.agent_tasks_enabled == configuration.enabled
                    and resolved_settings.agent_url == configuration.url
                    and resolved_settings.agent_api_key == configuration.api_key
                ):
                    return
                configured = await configure_global_agent(
                    resolved_settings,
                    agent_runtime_state,
                    agent_target_store,
                    hermes_factory,
                    agent_credential_resolver,
                    enabled=configuration.enabled,
                    url=configuration.url,
                    api_key=configuration.api_key,
                )
                app.state.agent_default_target_id = (
                    agent_runtime_state.default_target_id
                )
                app.state.agent_global_url = configuration.url
                app.state.agent_global_configured = configured.configured
                app.state.agent_global_credential_ref = configured.credential_ref
                app.state.agent_configuration_writable = configuration.enabled
                extension_service.capabilities_changed()

            def configure_agent_tuning(tuning: AgentTuning) -> None:
                hermes_connector.configure_timeouts(
                    request_seconds=tuning.http_timeout_seconds,
                    sse_read_seconds=tuning.sse_read_timeout_seconds,
                )
                if agent_run_processor is not None:
                    agent_run_processor.configure_scheduling(
                        concurrency=tuning.concurrency,
                        poll_interval_seconds=tuning.poll_seconds,
                    )
                if agent_run_supervisor is not None:
                    agent_run_supervisor.configure_poll_interval(tuning.poll_seconds)
                resolved_settings.agent_run_concurrency = tuning.concurrency
                resolved_settings.agent_run_poll_interval_seconds = tuning.poll_seconds
                resolved_settings.agent_http_timeout_seconds = (
                    tuning.http_timeout_seconds
                )
                resolved_settings.agent_sse_read_timeout_seconds = (
                    tuning.sse_read_timeout_seconds
                )
                resolved_settings.agent_reply_context_ttl_seconds = (
                    tuning.reply_context_ttl_seconds
                )

            def forget_agent_credentials() -> None:
                # An inactive integration may not have opened its persisted
                # credential file. Removal must still reach that stored copy.
                resolver = (
                    agent_credential_resolver
                    if agent_runtime_owned
                    else AgentCredentialResolver(
                        {},
                        credentials_file=Path(resolved_settings.agent_credentials_file),
                    )
                )
                resolver.clear_global_credentials()

            app.state.runtime_configuration = RuntimeConfigurationService(
                resolved_settings,
                model_runtime,
                custom_voice_client,
                configure_agent_tuning,
                forget_agent_credentials,
                configure_agent,
                codex_observer.configure if codex_observer is not None else None,
                experiment=experiment,
                speech_configured=schedule_local_tts_prefetch,
            )
            supervisors = [
                BackgroundTaskSupervisor(
                    llm_log_store,
                    name="llm-log-expiry",
                    poll_interval_seconds=LOG_CLEANUP_INTERVAL_SECONDS,
                )
            ]
            supervisors.append(
                BackgroundTaskSupervisor(
                    BackgroundMessageExpiryProcessor(background_message_store),
                    name="background-message-expiry-processor",
                    poll_interval_seconds=(
                        resolved_settings.background_poll_interval_seconds
                    ),
                )
            )
            supervisors.extend(
                [
                    BackgroundTaskSupervisor(
                        fact_processor,
                        name="fact-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                    BackgroundTaskSupervisor(
                        snapshot_processor,
                        name="dynamic-snapshot-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                    BackgroundTaskSupervisor(
                        static_profile_processor,
                        name="static-profile-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                    BackgroundTaskSupervisor(
                        dynamic_relation_processor,
                        name="dynamic-relation-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                    BackgroundTaskSupervisor(
                        static_relation_processor,
                        name="static-relation-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                    BackgroundTaskSupervisor(
                        mask_embedding_processor,
                        name="mask-embedding-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    ),
                ]
            )
            if agent_run_processor is not None:
                agent_run_supervisor = BackgroundTaskSupervisor(
                    agent_run_processor,
                    name="agent-run-processor",
                    poll_interval_seconds=(
                        resolved_settings.agent_run_poll_interval_seconds
                    ),
                )
                supervisors.append(agent_run_supervisor)
            if agent_message_processor is not None:
                supervisors.append(
                    BackgroundTaskSupervisor(
                        EnabledAgentProcessor(
                            agent_message_processor, agent_runtime_state
                        ),
                        name="agent-message-processor",
                        poll_interval_seconds=(
                            resolved_settings.background_poll_interval_seconds
                        ),
                    )
                )
            app.state.message_store = store
            app.state.message_attachments = message_attachments
            app.state.background_message_store = background_message_store
            app.state.fish_store = fish_store
            app.state.session_store = session_store
            app.state.agent_target_store = agent_target_store
            app.state.agent_task_store = agent_task_store
            app.state.agent_connector_registry = agent_connector_registry
            app.state.agent_connectors = dict(agent_connectors)
            app.state.agent_credential_resolver = agent_credential_resolver
            app.state.agent_runtime_state = agent_runtime_state
            app.state.agent_default_target_id = agent_default_target_id
            app.state.agent_global_url = agent_global_url
            app.state.agent_global_configured = agent_global_configured
            app.state.agent_global_credential_ref = agent_global_credential_ref
            app.state.agent_configuration_writable = agent_runtime_state.enabled
            if agent_task_route_service is not None:
                app.state.agent_task_route_service = agent_task_route_service
            if agent_reply_service is not None:
                app.state.agent_reply_service = agent_reply_service
            if agent_run_processor is not None:
                app.state.agent_run_processor = agent_run_processor
            if agent_message_processor is not None:
                app.state.agent_message_processor = agent_message_processor
            app.state.answer_coordinator = coordinator
            if voice_draft_coordinator is not None:
                app.state.voice_draft_coordinator = voice_draft_coordinator
            app.state.fact_store = fact_store
            app.state.recall_service = recall_service
            app.state.recall_store = recall_store
            app.state.embedding_space_store = embedding_space_store
            app.state.embedding_space = embedder.space
            app.state.performance_store = performance_store
            app.state.snapshot_store = snapshot_store
            app.state.memory_forget_store = MemoryForgetStore(pool)
            app.state.entity_store = entity_store
            app.state.dynamic_relation_store = dynamic_relation_store
            app.state.static_relation_store = static_relation_store
            app.state.settings = resolved_settings
            app.state.clock = app_clock
            app.state.pool = pool
            app.state.llm = llm
            app.state.reasoning_llm = reasoning_llm
            app.state.kws_service = kws_service
            app.state.pet_event_hub = pet_event_hub
            app.state.home_automation_service = home_service
            app.state.home_action_store = home_action_store
            app.state.observation_store = observation_store
            app.state.observation_service = observation_service
            app.state.computer_history_summary_service = (
                computer_history_summary_service
            )
            app.state.environment_frame_cache = environment_frame_cache
            app.state.environment_history_store = environment_history_store
            app.state.environment_history_service = environment_history_service
            app.state.request_environment_image_capability_probe = (
                request_environment_image_capability_probe
            )
            await asyncio.to_thread(release_unused_memory, "api-startup")
            for supervisor in supervisors:
                supervisor.start()
            if resolved_settings.desktop_model_routes is not None:
                schedule_local_tts_prefetch()
            yield
        finally:
            if local_tts_prefetch is not None:
                local_tts_prefetch.cancel()
                await asyncio.gather(local_tts_prefetch, return_exceptions=True)
            if image_capability_probe_task is not None:
                image_capability_probe_task.cancel()
                await asyncio.gather(
                    image_capability_probe_task,
                    return_exceptions=True,
                )
            for supervisor in reversed(supervisors):
                await supervisor.close()
            if codex_observer is not None:
                await codex_observer.close()
            if plugin_manager is not None:
                await plugin_manager.close()
            if extension_service is not None:
                await extension_service.close()
            if proactive_bridge is not None:
                await proactive_bridge.close()
            if experiment is not None:
                await experiment.close()
            if proactive_coordinator is not None:
                await proactive_coordinator.close()
            if agent_run_processor is not None:
                await agent_run_processor.close()
            if agent_message_processor is not None:
                await agent_message_processor.close()
            for connector in reversed(tuple(agent_connectors.values())):
                await connector.close()
            if voice_draft_coordinator is not None:
                await voice_draft_coordinator.close()
            if coordinator is not None:
                await coordinator.close()
            if home_service is not None:
                await home_service.close()
            if home_client is not None:
                await home_client.close()
            if kws_service is not None:
                await kws_service.close()
            forget_service_problems()
            await pet_event_hub.close()
            if custom_voice_client is not None:
                await custom_voice_client.close()
            if environment_history_service is not None:
                await environment_history_service.close()
            if session_history is not None:
                await session_history.close()
            if model_runtime is not None:
                await model_runtime.close()
            if call_journal is not None:
                await call_journal.close()
            if embedder is not None:
                await embedder.close()
            if pool is not None:
                await pool.close()
            await database.close()

    app = FastAPI(title="fish-memory", lifespan=lifespan)
    app.state.auth_service = auth_service
    app.state.pet_event_hub = pet_event_hub
    app.state.environment_frame_cache = environment_frame_cache
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=auth_service,
        scoped_auth=lambda scope: (
            (
                getattr(app.state, "plugin_manager", None) is not None
                and app.state.plugin_manager.authenticate_dev(scope)
            )
            or (
                scope.get("type") == "websocket"
                and scope.get("path") == "/v1/extensions/ws"
                and getattr(app.state, "extension_service", None) is not None
                and app.state.extension_service.authenticate(
                    list(scope.get("headers", []))
                )
                is not None
            )
        ),
    )
    app.include_router(auth_router)
    app.include_router(catalog_router)
    app.include_router(internal_router)
    app.include_router(router)
    app.include_router(agent_router)
    app.include_router(agent_target_router)
    app.include_router(background_message_router)
    app.include_router(proactive_router)
    app.include_router(extension_router)
    app.include_router(runtime_configuration_router)
    app.include_router(llm_log_router)
    app.include_router(gateway_router)
    app.include_router(tts_internal_router)
    app.include_router(tts_router)
    app.include_router(kws_router)
    app.include_router(home_router)
    app.include_router(pet_router)
    app.include_router(observation_internal_router)
    app.include_router(environment_history_router)
    app.include_router(history_router)
    app.include_router(observation_router)
    app.include_router(environment_frame_router)

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/internal/model-image-input", include_in_schema=False)
    async def internal_model_image_input() -> dict[str, str]:
        """Whether 对话大脑 and 推理大脑 read images; unknown ones are checked."""
        chat = getattr(app.state, "llm", None)
        reasoning = getattr(app.state, "reasoning_llm", None)
        results = await asyncio.gather(
            image_input_capability(chat), image_input_capability(reasoning)
        )
        return {"chat": results[0], "reasoning": results[1]}

    @app.get("/internal/capabilities", include_in_schema=False)
    async def internal_capabilities() -> dict[str, bool]:
        return {
            "voiceDrafts": resolved_settings.companion_voice_draft_enabled,
        }

    @app.get("/ready", tags=["operations"])
    async def ready() -> JSONResponse:
        if getattr(app.state, "pool", None) is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready"},
            )
        return JSONResponse(content={"status": "ready"})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request
        details = error.errors()
        message = details[0]["msg"] if details else "invalid request"
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": message,
                }
            },
        )

    return app
