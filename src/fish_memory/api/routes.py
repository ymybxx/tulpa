import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from time import perf_counter_ns
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from fish_memory import failures
from fish_memory.agent.models import AgentMessageSubjectKind
from fish_memory.agent.replies import (
    AgentReplyClassificationError,
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyService,
    reply_context_fingerprint,
)
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.sse import AnswerEvent, format_sse_event, iter_answer_events
from fish_memory.answer.voice_draft_types import (
    VoiceDraftKey,
    VoiceDraftState,
    voice_answer_message_id,
)
from fish_memory.answer.voice_drafts import (
    VoiceDraftCoordinator,
    VoiceDraftCoordinatorClosed,
    VoiceDraftLease,
    VoiceDraftRevisionConflict,
)
from fish_memory.api.models import (
    AskRequest,
    AskResponse,
    AssistantFishResponse,
    BackgroundProgressResponse,
    BackgroundTaskProgressResponse,
    EntityFactPageResponse,
    EntityFactResponse,
    EntitySummaryResponse,
    FishCreateRequest,
    FishCreateResponse,
    FishListResponse,
    FishManagementResponse,
    FishSummaryResponse,
    FishVoiceConfigRequest,
    FishVoiceConfigResponse,
    FishVoiceSelectionResponse,
    HomeActionDebugResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MessageAttachmentResponse,
    MessageListResponse,
    MessageOriginResponse,
    MessageResponse,
    PerformanceStepListResponse,
    PerformanceStepResponse,
    PetVoicePerformanceRequest,
    ProgressRequest,
    RecallFactResponse,
    RecallPlanResponse,
    RecallRequest,
    RecallResponse,
    RecallTimePlanResponse,
    RecallTraceResponse,
    ReplyContextRequest,
    SessionListResponse,
    SessionResponse,
    SnapshotItemResponse,
    SnapshotOverviewResponse,
    SoulProfileResponse,
    SoulUpdateRequest,
    TtsPerformanceRequest,
    VoiceDraftCancelRequest,
    VoiceDraftStatusResponse,
    VoiceDraftUpsertRequest,
    VoiceTurnCommitRequest,
    WakeWordsRequest,
    WakeWordsResponse,
)
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.db.entities import EntityStore
from fish_memory.db.facts import FactStore
from fish_memory.db.fishes import FishStore
from fish_memory.db.home_actions import HomeActionStore
from fish_memory.db.memory_forget import MemoryForgetStore
from fish_memory.db.message_attachments import (
    MAX_TEXT_CHARACTERS,
    AttachmentError,
    MessageAttachmentStore,
)
from fish_memory.db.messages import (
    FishDeletedError,
    LegacyFishReadOnlyError,
    MessageIdentityConflictError,
    MessageOrigin,
    MessagesStore,
)
from fish_memory.db.performance import PerformanceStore
from fish_memory.db.sessions import SessionNotFoundError, SessionStore
from fish_memory.db.snapshots import SnapshotStore
from fish_memory.models import (
    Message,
    MessageStatus,
    Role,
    Session,
    visible_message_content,
)
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.hub import PetEventHub
from fish_memory.recall.service import FactRecallService, RecallEmbeddingError
from fish_memory.snapshots.models import NormalizedSnapshotOverview
from fish_memory.soul import SoulProfile
from fish_memory.tts.catalog import get_local_tts_catalog, get_tts_catalog
from fish_memory.tts.local import voice_provider
from fish_memory.tts.voice_config import (
    FishVoiceConfig,
    VoiceKind,
    is_supported_voice,
    resolve_voice_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/fishes/{fish_id}", tags=["public"])
catalog_router = APIRouter(prefix="/v1/fishes", tags=["public"])
internal_router = APIRouter(
    prefix="/internal/fishes/{fish_id}",
    tags=["internal"],
    include_in_schema=False,
)


async def _resolve_session(
    request: Request,
    fish_id: UUID,
    session_id: UUID | None,
) -> Session | None:
    store = cast(SessionStore, request.app.state.session_store)
    try:
        if session_id is not None:
            return await store.require_owned(fish_id, session_id)
        return await store.get_latest(fish_id)
    except SessionNotFoundError:
        return None


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@catalog_router.get("", response_model=FishListResponse, response_model_by_alias=True)
async def list_fishes(request: Request) -> FishListResponse:
    store = cast(MessagesStore, request.app.state.message_store)
    fishes = await store.list_fishes()
    return FishListResponse(
        items=[
            FishSummaryResponse(
                fish_id=fish.fish_id,
                last_message=fish.last_message,
                last_active_at=fish.last_active_at,
                message_count=fish.message_count,
                is_assistant=fish.is_assistant,
                name=fish.name,
                role=fish.role,
                evaluation=fish.evaluation,
            )
            for fish in fishes
        ]
    )


@catalog_router.get(
    "/assistant",
    response_model=AssistantFishResponse,
    response_model_by_alias=True,
)
async def get_assistant_fish(request: Request) -> AssistantFishResponse:
    store = cast(FishStore, request.app.state.fish_store)
    return AssistantFishResponse(fish_id=await store.get_assistant())


@catalog_router.post(
    "",
    response_model=FishCreateResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_fish(
    body: FishCreateRequest,
    request: Request,
) -> FishCreateResponse:
    store = cast(FishStore, request.app.state.fish_store)
    fish_id = await store.create(
        role=body.role,
        setting=body.setting,
        name=body.name,
        background=body.background,
    )
    await _bind_desktop_assistant_if_missing(request, fish_id)
    return FishCreateResponse(fish_id=fish_id)


async def _bind_desktop_assistant_if_missing(
    request: Request,
    fish_id: UUID,
) -> None:
    settings = getattr(request.app.state, "settings", None)
    if not getattr(settings, "desktop_mode", False):
        return
    store = cast(FishStore, request.app.state.fish_store)
    if await store.get_assistant() is None:
        if await store.bind_assistant(fish_id):
            _notify_assistant_binding_changed(request)


def _notify_assistant_binding_changed(request: Request) -> None:
    """Invalidate desktop capture targets; clients re-read the persisted binding."""
    hub = cast(PetEventHub | None, getattr(request.app.state, "pet_event_hub", None))
    if hub is not None:
        hub.publish_control("pet.assistant.changed", None, {})


@catalog_router.put(
    "/{fish_id}/assistant",
    response_model=AssistantFishResponse,
    response_model_by_alias=True,
)
async def bind_assistant_fish(
    fish_id: UUID,
    request: Request,
) -> AssistantFishResponse | JSONResponse:
    store = cast(FishStore, request.app.state.fish_store)
    if not await store.bind_assistant(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    _notify_assistant_binding_changed(request)
    return AssistantFishResponse(fish_id=fish_id)


@catalog_router.post(
    "/{fish_id}/birth",
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def announce_fish_birth(fish_id: UUID, request: Request) -> Response:
    """The desktop calls this once a new character is up, so TA speaks first."""
    from fish_memory.birth import announce_birth

    events = getattr(request.app.state, "extension_service", None)
    if events is None:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE, "events_unavailable", "事件服务未启动"
        )
    delivered = await announce_birth(
        events,
        cast(FishStore, request.app.state.fish_store),
        cast(SessionStore, request.app.state.session_store),
        fish_id,
    )
    if not delivered:
        return error_response(
            status.HTTP_409_CONFLICT, "not_assistant", "只有当前角色能开口"
        )
    return Response(status_code=status.HTTP_202_ACCEPTED)


@catalog_router.delete("/assistant", status_code=status.HTTP_204_NO_CONTENT)
async def unbind_assistant_fish(request: Request) -> Response:
    store = cast(FishStore, request.app.state.fish_store)
    await store.unbind_assistant()
    _notify_assistant_binding_changed(request)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@catalog_router.delete("/{fish_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_fish(
    fish_id: UUID,
    request: Request,
) -> Response:
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    if not await coordinator.delete_fish(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    _notify_assistant_binding_changed(request)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _snapshot_overview_response(
    overview: NormalizedSnapshotOverview,
) -> SnapshotOverviewResponse:
    return SnapshotOverviewResponse(
        snapshot_type=overview.snapshot_type,
        version=overview.version,
        updated_at=overview.updated_at,
        through_fact_sequence=overview.through_fact_sequence,
        items=[
            SnapshotItemResponse(
                text=item.summary,
                id=item.id,
                last_supported_at=item.last_supported_at,
                source_fact_count=len(item.support_fact_ids),
            )
            for item in overview.items
        ],
    )


async def _fish_management_response(
    fish_id: UUID,
    request: Request,
) -> FishManagementResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.exists(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    profile = await fish_store.get_profile(fish_id)
    if profile is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    settings = getattr(request.app.state, "settings", None)
    base_prompt = getattr(settings, "system_prompt", "")
    effective_soul = profile.render(base_prompt)
    snapshot_store = cast(SnapshotStore, request.app.state.snapshot_store)
    entity_store = cast(EntityStore, request.app.state.entity_store)
    overviews = await snapshot_store.list_normalized_current_overviews(fish_id)
    entities = await entity_store.list_summaries(fish_id)
    voice_config = await fish_store.get_voice_config(fish_id)
    by_type = {overview.snapshot_type: overview for overview in overviews}
    return FishManagementResponse(
        fish_id=fish_id,
        is_assistant=await fish_store.is_assistant(fish_id),
        soul_profile=SoulProfileResponse(
            role=profile.role,
            setting=profile.setting,
            name=profile.name,
            background=profile.background,
        ),
        effective_soul=effective_soul,
        voice_config=(
            None
            if voice_config is None
            else FishVoiceConfigResponse.model_validate(voice_config.model_dump())
        ),
        dynamic_snapshot=(
            None
            if by_type.get("dynamic") is None
            else _snapshot_overview_response(by_type["dynamic"])
        ),
        static_profile=(
            None
            if by_type.get("static") is None
            else _snapshot_overview_response(by_type["static"])
        ),
        entities=[
            EntitySummaryResponse(
                entity_id=entity.entity_id,
                aliases=list(entity.aliases),
                fact_count=entity.fact_count,
            )
            for entity in entities
        ],
        created_at=await fish_store.get_created_at(fish_id),
    )


@internal_router.get(
    "/management",
    response_model=FishManagementResponse,
    response_model_by_alias=True,
)
async def get_fish_management(
    fish_id: UUID,
    request: Request,
) -> FishManagementResponse | JSONResponse:
    return await _fish_management_response(fish_id, request)


@router.patch(
    "/soul",
    response_model=SoulProfileResponse,
    response_model_by_alias=True,
)
async def update_fish_soul(
    fish_id: UUID,
    body: SoulUpdateRequest,
    request: Request,
) -> SoulProfileResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    profile = SoulProfile.from_values(
        body.role, body.setting, body.name, body.background
    )
    if not await fish_store.update_profile(fish_id, profile):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    proactive = getattr(request.app.state, "proactive_coordinator", None)
    if proactive is not None:
        proactive.profile_changed(fish_id)
    return SoulProfileResponse(
        role=profile.role,
        setting=profile.setting,
        name=profile.name,
        background=profile.background,
    )


@router.get(
    "/soul",
    response_model=SoulProfileResponse,
    response_model_by_alias=True,
)
async def get_fish_soul(
    fish_id: UUID,
    request: Request,
) -> SoulProfileResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    profile = await fish_store.get_profile(fish_id)
    if profile is None:
        return error_response(404, "fish_not_found", "fish 不存在")
    return SoulProfileResponse(
        role=profile.role,
        setting=profile.setting,
        name=profile.name,
        background=profile.background,
    )


@router.get(
    "/wake-words",
    response_model=WakeWordsResponse,
    response_model_by_alias=True,
)
async def get_fish_wake_words(
    fish_id: UUID,
    request: Request,
) -> WakeWordsResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.exists(fish_id):
        return error_response(404, "fish_not_found", "fish 不存在")
    words = await fish_store.get_wake_words(fish_id)
    if words is not None:
        return WakeWordsResponse(words=list(words), customized=True)
    # A Fish that never chose its own keeps answering to the app-wide word.
    fallback = getattr(request.app.state.settings, "wake_word", "小鱼")
    return WakeWordsResponse(words=[fallback], customized=False)


@router.put(
    "/wake-words",
    response_model=WakeWordsResponse,
    response_model_by_alias=True,
)
async def update_fish_wake_words(
    fish_id: UUID,
    body: WakeWordsRequest,
    request: Request,
) -> WakeWordsResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.update_wake_words(fish_id, body.words):
        return error_response(404, "fish_not_found", "fish 不存在")
    return WakeWordsResponse(words=body.words, customized=True)


@router.post(
    "/memory/forget-preview",
    response_model=MemoryForgetResponse,
    response_model_by_alias=True,
)
async def preview_forget_memory(
    fish_id: UUID,
    body: MemoryForgetRequest,
    request: Request,
) -> MemoryForgetResponse | JSONResponse:
    store = cast(MemoryForgetStore, request.app.state.memory_forget_store)
    plan = await store.preview(fish_id, item_id=body.item_id, fact_id=body.fact_id)
    if plan is None:
        return error_response(404, "memory_not_found", "这条记忆已经不在了")
    return MemoryForgetResponse(fact_count=plan.fact_count, item_count=plan.item_count)


@router.post(
    "/memory/forget",
    response_model=MemoryForgetResponse,
    response_model_by_alias=True,
)
async def forget_memory(
    fish_id: UUID,
    body: MemoryForgetRequest,
    request: Request,
) -> MemoryForgetResponse | JSONResponse:
    store = cast(MemoryForgetStore, request.app.state.memory_forget_store)
    plan = await store.forget(fish_id, item_id=body.item_id, fact_id=body.fact_id)
    if plan is None:
        return error_response(404, "memory_not_found", "这条记忆已经不在了")
    return MemoryForgetResponse(fact_count=plan.fact_count, item_count=plan.item_count)


@router.patch(
    "/voice",
    response_model=FishVoiceConfigResponse,
    response_model_by_alias=True,
)
async def update_fish_voice(
    fish_id: UUID,
    body: FishVoiceConfigRequest,
    request: Request,
) -> FishVoiceConfigResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    config = FishVoiceConfig.model_validate(body.model_dump())
    catalog = (*get_tts_catalog(), *get_local_tts_catalog())
    if not is_supported_voice(config, catalog):
        return error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "unsupported_voice",
            "音色与模型不匹配",
        )
    if not await fish_store.update_voice_config(fish_id, config):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    if config.kind is VoiceKind.LOCAL and (
        prepare := getattr(request.app.state, "prepare_local_voice", None)
    ):
        # The local voice may not be downloaded yet when the default is 通义.
        prepare()
    return FishVoiceConfigResponse.model_validate(config.model_dump())


@router.get(
    "/voice",
    response_model=FishVoiceSelectionResponse,
    response_model_by_alias=True,
)
async def get_fish_voice(
    fish_id: UUID,
    request: Request,
) -> FishVoiceSelectionResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.exists(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    settings = request.app.state.settings
    saved = await fish_store.get_voice_config(fish_id)
    # Each character picks 通义 or 本机 through its own voice.
    provider = voice_provider(saved, settings)
    resolved = resolve_voice_config(saved, provider=provider)
    return FishVoiceSelectionResponse(
        provider=provider,
        config=FishVoiceConfigResponse.model_validate(resolved.config.model_dump()),
        used_fallback=resolved.used_fallback,
    )


@internal_router.get(
    "/entities/{entity_id}/facts",
    response_model=EntityFactPageResponse,
    response_model_by_alias=True,
)
async def list_entity_facts(
    fish_id: UUID,
    entity_id: UUID,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> EntityFactPageResponse | JSONResponse:
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.exists(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    entity_store = cast(EntityStore, request.app.state.entity_store)
    page = await entity_store.list_facts(
        fish_id,
        entity_id,
        limit=limit,
        offset=offset,
    )
    return EntityFactPageResponse(
        items=[
            EntityFactResponse(
                fact_id=fact.id,
                text=fact.text,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                mentioned_at=fact.mentioned_at,
                source_role=fact.source_role,
                invalidated_at=fact.invalidated_at,
            )
            for fact in page.items
        ],
        has_more=page.has_more,
    )


@catalog_router.post(
    "/{fish_id}/sessions",
    response_model=SessionResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    fish_id: UUID,
    request: Request,
) -> SessionResponse | JSONResponse:
    store = cast(SessionStore, request.app.state.session_store)
    try:
        session = await store.create(fish_id)
    except SessionNotFoundError:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )
    return SessionResponse(
        session_id=session.id,
        created_at=session.created_at,
        last_message_at=session.last_message_at,
    )


@catalog_router.get(
    "/{fish_id}/sessions",
    response_model=SessionListResponse,
    response_model_by_alias=True,
)
async def list_sessions(
    fish_id: UUID,
    request: Request,
) -> SessionListResponse:
    store = cast(SessionStore, request.app.state.session_store)
    sessions = await store.list(fish_id)
    return SessionListResponse(
        items=[
            SessionResponse(
                session_id=session.id,
                created_at=session.created_at,
                last_message_at=session.last_message_at,
            )
            for session in sessions
        ]
    )


@catalog_router.get(
    "/{fish_id}/messages",
    response_model=MessageListResponse,
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def list_messages(
    fish_id: UUID,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
    before: int | None = Query(default=None, ge=0),
    after: int | None = Query(default=None, ge=0),
    session_id: Annotated[UUID | None, Query(alias="sessionId")] = None,
) -> MessageListResponse | JSONResponse:
    store = cast(MessagesStore, request.app.state.message_store)
    session = await _resolve_session(request, fish_id, session_id)
    if session is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    if hasattr(store, "list_message_page"):
        page = await store.list_message_page(
            fish_id,
            session.id,
            limit=limit,
            before=before,
            after=after,
        )
    else:
        from fish_memory.db.messages import MessagePage

        page = MessagePage(
            items=await store.list_messages(fish_id, session.id, limit=limit),
            has_before=False,
            has_after=False,
            min_sequence=None,
            max_sequence=None,
        )
    return MessageListResponse(
        items=[
            MessageResponse(
                message_id=message.id,
                sequence=message.sequence,
                role=message.role,
                content=visible_message_content(message),
                status=message.status,
                consumed_offset=message.consumed_offset,
                created_at=message.created_at,
                finished_at=message.finished_at,
                failure=message.failure,
            )
            for message in page.items
        ],
        has_before=(
            page.has_before if before is not None or after is not None else None
        ),
        has_after=(page.has_after if before is not None or after is not None else None),
        min_sequence=(
            page.min_sequence if before is not None or after is not None else None
        ),
        max_sequence=(
            page.max_sequence if before is not None or after is not None else None
        ),
    )


def _origin_response(origin: MessageOrigin) -> MessageOriginResponse:
    if origin.source == "agent":
        return MessageOriginResponse(kind="task")
    if origin.source == "plugin" or origin.observation_type == "plugin.event":
        return MessageOriginResponse(kind="plugin")
    if origin.observation_type is None:
        return MessageOriginResponse(kind="self")
    kind = "sense" if origin.observation_source == "environment_history" else "event"
    return MessageOriginResponse(kind=kind, app_name=origin.app_name)


async def _remember_input_mode(
    store: MessagesStore, message: Message, mode: str | None, inserted: bool
) -> None:
    """Best effort: the chat page's typed/spoken mark must never fail a turn."""
    if mode is None or not inserted or message.role is not Role.USER:
        return
    try:
        await store.record_input_mode(message.id, mode)
    except Exception:
        logger.warning("Could not record how message %s was entered", message.id)


async def _timeline_responses(
    store: MessagesStore,
    fish_id: UUID,
    messages: list[Message],
    attachments: MessageAttachmentStore | None = None,
) -> list[MessageResponse]:
    origins = await store.message_origins(
        fish_id, [message.id for message in messages if message.role == "assistant"]
    )
    modes = await store.input_modes(
        fish_id, [message.id for message in messages if message.role == "user"]
    )
    unsaid = await store.unsaid_texts(
        fish_id,
        [
            message.id
            for message in messages
            if message.status is MessageStatus.INTERRUPTED
        ],
    )
    carried = (
        await attachments.names_for_messages(
            [message.id for message in messages if message.role == "user"]
        )
        if attachments is not None
        else {}
    )
    responses = []
    for message in messages:
        response = _message_response(message)
        if items := carried.get(message.id):
            response.attachments = [
                MessageAttachmentResponse.model_validate(item) for item in items
            ]
        if (origin := origins.get(message.id)) is not None:
            response.origin = _origin_response(origin)
        if (mode := modes.get(message.id)) is not None:
            response.input_mode = mode
        response.unsaid = unsaid.get(message.id)
        response.session_id = message.session_id
        responses.append(response)
    return responses


def _message_response(message: Message) -> MessageResponse:
    return MessageResponse(
        message_id=message.id,
        sequence=message.sequence,
        role=message.role,
        content=visible_message_content(message),
        status=message.status,
        consumed_offset=message.consumed_offset,
        created_at=message.created_at,
        finished_at=message.finished_at,
        failure=message.failure,
    )


@catalog_router.get(
    "/{fish_id}/timeline",
    response_model=MessageListResponse,
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def list_timeline(
    fish_id: UUID,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    before: int | None = Query(default=None, ge=0),
    after: int | None = Query(default=None, ge=0),
) -> MessageListResponse | JSONResponse:
    """The whole conversation with a fish, across sessions, for the chat page."""
    if before is not None and after is not None:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_cursor",
            "before 和 after 不能同时使用",
        )
    store = cast(MessagesStore, request.app.state.message_store)
    page = await store.list_timeline_page(
        fish_id, limit=limit, before=before, after=after
    )
    return MessageListResponse(
        items=await _timeline_responses(
            store,
            fish_id,
            page.items,
            getattr(request.app.state, "message_attachments", None),
        ),
        has_before=page.has_before,
        has_after=page.has_after,
        min_sequence=page.min_sequence,
        max_sequence=page.max_sequence,
    )


@catalog_router.get(
    "/{fish_id}/messages/search",
    response_model=MessageListResponse,
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def search_messages(
    fish_id: UUID,
    request: Request,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
) -> MessageListResponse:
    store = cast(MessagesStore, request.app.state.message_store)
    query = q.strip()
    found = await store.search_messages(fish_id, query, limit=limit) if query else []
    return MessageListResponse(items=await _timeline_responses(store, fish_id, found))


@internal_router.get(
    "/messages/{message_id}/performance",
    response_model=PerformanceStepListResponse,
    response_model_by_alias=True,
)
async def list_message_performance(
    fish_id: UUID,
    message_id: UUID,
    request: Request,
) -> PerformanceStepListResponse:
    store = cast(PerformanceStore, request.app.state.performance_store)
    home_store = cast(HomeActionStore, request.app.state.home_action_store)
    steps, home_actions = await asyncio.gather(
        store.list_message_steps(fish_id, message_id),
        home_store.list_message_runs(fish_id, message_id),
    )
    return PerformanceStepListResponse(
        items=[
            PerformanceStepResponse(
                step_name=step.name,
                step_index=step.index,
                started_offset_us=step.started_offset_us,
                duration_us=step.duration_us,
                input_payload=step.input_payload,
                output_payload=step.output_payload,
                error=step.error,
            )
            for step in steps
        ],
        home_actions=[
            HomeActionDebugResponse(
                id=action.id,
                status=action.status.value,
                action=dict(action.action),
                entity_id=action.entity_id,
                service_domain=action.service_domain,
                service_name=action.service_name,
                accepted=action.accepted,
                confirmed=action.confirmed,
                result=cast(JsonValue, action.result),
                error=action.error,
                created_at=action.created_at,
                updated_at=action.updated_at,
                finished_at=action.finished_at,
            )
            for action in home_actions
        ],
    )


@internal_router.post(
    "/messages/{message_id}/tts-performance",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def persist_tts_performance(
    fish_id: UUID,
    message_id: UUID,
    payload: TtsPerformanceRequest,
    request: Request,
) -> Response:
    store = cast(PerformanceStore, request.app.state.performance_store)
    stored = await store.replace_tts_metrics(fish_id, message_id, payload.metrics)
    if not stored:
        return error_response(404, "message_not_found", "message not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@internal_router.post(
    "/messages/{message_id}/pet-voice-performance",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def persist_pet_voice_performance(
    fish_id: UUID,
    message_id: UUID,
    payload: PetVoicePerformanceRequest,
    request: Request,
) -> Response:
    if payload.metrics.answer_message_id != message_id:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "answer_message_mismatch",
            "answerMessageId must match the message path",
        )
    metrics = payload.metrics.model_dump(
        mode="json",
        by_alias=True,
        exclude_unset=True,
    )
    store = cast(PerformanceStore, request.app.state.performance_store)
    stored = await store.replace_pet_voice_metrics(fish_id, message_id, metrics)
    if not stored:
        return error_response(404, "message_not_found", "message not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class MessageFailureRequest(BaseModel):
    failure: str = Field(max_length=32)


@internal_router.post(
    "/messages/{message_id}/failure",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def record_message_failure(
    fish_id: UUID,
    message_id: UUID,
    payload: MessageFailureRequest,
    request: Request,
) -> Response:
    failure = failures.ServiceFailure.parse(payload.failure)
    if failure is None:
        return error_response(400, "invalid_failure", "unknown failure")
    store = cast(MessagesStore, request.app.state.message_store)
    await store.record_failure(message_id, fish_id, failure.code)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@internal_router.get(
    "/background-progress",
    response_model=BackgroundProgressResponse,
    response_model_by_alias=True,
)
async def get_background_progress(
    fish_id: UUID,
    request: Request,
    session_id: Annotated[UUID | None, Query(alias="sessionId")] = None,
) -> BackgroundProgressResponse | JSONResponse:
    session = await _resolve_session(request, fish_id, session_id)
    if session is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    store = cast(FactStore, request.app.state.fact_store)
    progress = await store.get_progress(fish_id, session.id)
    return BackgroundProgressResponse(
        tasks=[
            BackgroundTaskProgressResponse(
                name="fact_extraction",
                through_sequence=progress.through_sequence,
                target_sequence=progress.target_sequence,
                pending=progress.pending,
                last_error=progress.last_error,
                updated_at=progress.updated_at,
            )
        ]
    )


def _voice_draft_coordinator(request: Request) -> VoiceDraftCoordinator | None:
    coordinator = getattr(request.app.state, "voice_draft_coordinator", None)
    return cast(VoiceDraftCoordinator | None, coordinator)


def _agent_reply_service(request: Request) -> AgentReplyService | None:
    service = getattr(request.app.state, "agent_reply_service", None)
    return cast(AgentReplyService | None, service)


def _freeze_reply_context(
    context: ReplyContextRequest | None,
) -> AgentReplyContext | None:
    if context is None:
        return None
    return AgentReplyContext(AgentMessageSubjectKind(context.kind), context.id)


async def _require_voice_session(
    request: Request,
    fish_id: UUID,
    session_id: UUID,
) -> JSONResponse | None:
    store = cast(SessionStore, request.app.state.session_store)
    try:
        await store.require_owned(fish_id, session_id)
    except SessionNotFoundError:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    return None


async def _settle_voice_lease(lease: VoiceDraftLease | None) -> None:
    if lease is not None:
        await lease.prepared.cancel()


@internal_router.put(
    "/sessions/{session_id}/voice-turns/{turn_id}/draft",
    response_model=VoiceDraftStatusResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upsert_voice_draft(
    fish_id: UUID,
    session_id: UUID,
    turn_id: UUID,
    body: VoiceDraftUpsertRequest,
    request: Request,
) -> VoiceDraftStatusResponse | JSONResponse:
    if not body.content.strip():
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_content",
            "content 必须是非空字符串",
        )
    invalid_session = await _require_voice_session(request, fish_id, session_id)
    if invalid_session is not None:
        return invalid_session
    drafts = _voice_draft_coordinator(request)
    settings = getattr(request.app.state, "settings", None)
    if drafts is None or (
        settings is not None and not settings.companion_voice_draft_enabled
    ):
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "voice_drafts_disabled",
            "voice draft preflight 未启用",
        )
    key = VoiceDraftKey(fish_id, session_id, turn_id)
    try:
        snapshot = await drafts.upsert(
            key,
            revision=body.revision,
            content=body.content,
            reply_context=_freeze_reply_context(body.reply_context),
            **(
                {
                    "speech_started_at": body.speech_started_at,
                    "frame_snapshot_at": body.frame_snapshot_at,
                }
                if body.speech_started_at is not None
                and body.frame_snapshot_at is not None
                else {}
            ),
        )
    except VoiceDraftRevisionConflict:
        return error_response(
            status.HTTP_409_CONFLICT,
            "revision_conflict",
            "相同 revision 的候选内容不一致",
        )
    if snapshot is None:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "voice_draft_unavailable",
            "voice draft preflight 暂不可用",
        )
    return VoiceDraftStatusResponse(
        turn_id=turn_id,
        revision=snapshot.revision,
        content=snapshot.content,
        state=snapshot.state.value,
    )


@internal_router.post(
    "/sessions/{session_id}/voice-turns/{turn_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def cancel_voice_draft(
    fish_id: UUID,
    session_id: UUID,
    turn_id: UUID,
    body: VoiceDraftCancelRequest,
    request: Request,
) -> Response:
    invalid_session = await _require_voice_session(request, fish_id, session_id)
    if invalid_session is not None:
        return invalid_session
    drafts = _voice_draft_coordinator(request)
    if drafts is None:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "voice_drafts_disabled",
            "voice draft preflight 未启用",
        )
    await drafts.cancel(
        VoiceDraftKey(fish_id, session_id, turn_id),
        revision=body.revision,
        state=VoiceDraftState.CANCELLED,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@internal_router.get(
    "/sessions/{session_id}/voice-turns/{turn_id}/events",
    response_model=None,
)
async def voice_draft_events(
    fish_id: UUID,
    session_id: UUID,
    turn_id: UUID,
    request: Request,
    revision: int = Query(ge=1),
) -> StreamingResponse | JSONResponse:
    invalid_session = await _require_voice_session(request, fish_id, session_id)
    if invalid_session is not None:
        return invalid_session
    drafts = _voice_draft_coordinator(request)
    if drafts is None:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "voice_drafts_disabled",
            "voice draft preflight 未启用",
        )
    key = VoiceDraftKey(fish_id, session_id, turn_id)

    async def stream() -> AsyncIterator[str]:
        async for event in drafts.iter_events(key, revision=revision):
            yield format_sse_event(
                AnswerEvent(
                    event=event.event,
                    event_id=event.event_id,
                    data=event.data,
                )
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@internal_router.post(
    "/sessions/{session_id}/voice-turns/{turn_id}/commit",
    response_model=AskResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def commit_voice_turn(
    fish_id: UUID,
    session_id: UUID,
    turn_id: UUID,
    body: VoiceTurnCommitRequest,
    request: Request,
) -> AskResponse | JSONResponse:
    request_started_ns = perf_counter_ns()
    reply_context = _freeze_reply_context(body.reply_context)
    context_fingerprint = reply_context_fingerprint(reply_context)
    if not body.content.strip():
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_content",
            "content 必须是非空字符串",
        )
    invalid_session = await _require_voice_session(request, fish_id, session_id)
    if invalid_session is not None:
        return invalid_session

    key = VoiceDraftKey(fish_id, session_id, turn_id)
    drafts = _voice_draft_coordinator(request)
    if drafts is None:
        return await _commit_voice_turn_authoritative(
            fish_id=fish_id,
            session_id=session_id,
            turn_id=turn_id,
            body=body,
            request=request,
            request_started_ns=request_started_ns,
            reply_context=reply_context,
            context_fingerprint=context_fingerprint,
            key=key,
            drafts=None,
        )
    try:
        async with drafts.commit_guard(key):
            return await _commit_voice_turn_authoritative(
                fish_id=fish_id,
                session_id=session_id,
                turn_id=turn_id,
                body=body,
                request=request,
                request_started_ns=request_started_ns,
                reply_context=reply_context,
                context_fingerprint=context_fingerprint,
                key=key,
                drafts=drafts,
            )
    except VoiceDraftCoordinatorClosed:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "voice_draft_unavailable",
            "voice draft coordinator 已关闭",
        )


async def _commit_voice_turn_authoritative(
    *,
    fish_id: UUID,
    session_id: UUID,
    turn_id: UUID,
    body: VoiceTurnCommitRequest,
    request: Request,
    request_started_ns: int,
    reply_context: AgentReplyContext | None,
    context_fingerprint: str,
    key: VoiceDraftKey,
    drafts: VoiceDraftCoordinator | None,
) -> AskResponse | JSONResponse:
    try:
        lease = (
            None
            if drafts is None
            else await drafts.finalize(
                key,
                revision=body.revision,
                promote_revision=body.promote_revision,
                content=body.content,
                context_fingerprint=context_fingerprint,
            )
        )
    except VoiceDraftRevisionConflict:
        return error_response(
            status.HTTP_409_CONFLICT,
            "voice_turn_conflict",
            "turn_id 已用于不同的 authoritative content 或 replyContext",
        )
    reply_service = _agent_reply_service(request)
    if reply_context is not None and reply_service is None:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "agent_replies_unavailable",
            "Agent 精确回复服务未启用",
        )
    store = cast(MessagesStore, request.app.state.message_store)
    persist_started_ns = perf_counter_ns()
    handled = False
    inserted = True
    try:
        if reply_context is None:
            message = await store.insert_user_message(
                fish_id,
                body.content,
                session_id=session_id,
                created_at=None if lease is None else lease.created_at,
                message_id=turn_id,
            )
        else:
            assert reply_service is not None
            prepared_reply = (
                None
                if lease is None
                else getattr(lease.prepared, "prepared_reply", None)
            )
            if prepared_reply is not None:
                assert lease is not None
                result = await reply_service.commit_prepared_reply(
                    prepared_reply,
                    message_id=turn_id,
                    created_at=lease.created_at,
                )
            else:
                result = await reply_service.handle(
                    fish_id,
                    session_id,
                    body.content,
                    reply_context,
                    message_id=turn_id,
                    created_at=None if lease is None else lease.created_at,
                )
            message = result.message
            handled = result.handled
            inserted = result.inserted
    except AgentReplyClassificationError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "reply_classification_failed",
            "Agent 回复分类失败，未消费 replyContext",
        )
    except AgentReplyConflictError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_409_CONFLICT,
            "reply_context_conflict",
            "replyContext 已过期、已消费或不属于当前会话",
        )
    except FishDeletedError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_410_GONE,
            "fish_deleted",
            "fish 已删除",
        )
    except LegacyFishReadOnlyError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_409_CONFLICT,
            "legacy_fish_read_only",
            "legacy fish 仅支持查看，不能继续对话",
        )
    except EmbeddingSpaceMismatchError as error:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_409_CONFLICT,
            "embedding_space_mismatch",
            str(error),
        )
    except SessionNotFoundError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    except MessageIdentityConflictError:
        await _settle_voice_lease(lease)
        return error_response(
            status.HTTP_409_CONFLICT,
            "voice_turn_conflict",
            "turn_id 已用于不同的 authoritative content",
        )
    except BaseException:
        await _settle_voice_lease(lease)
        raise

    try:
        await _bind_desktop_assistant_if_missing(request, fish_id)
    except BaseException:
        await _settle_voice_lease(lease)
        raise
    await _remember_input_mode(store, message, "spoken", inserted)
    persist_finished_ns = perf_counter_ns()
    tracker = MessagePerformanceTracker(origin_ns=request_started_ns)
    tracker.record(
        "ask.persist_message",
        started_ns=persist_started_ns,
        finished_ns=persist_finished_ns,
    )
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    if handled or not inserted:
        await _settle_voice_lease(lease)
        return AskResponse(
            message_id=message.id,
            session_id=message.session_id,
            sequence=message.sequence,
            status=message.status,
            created_at=message.created_at,
            answer_expected=not handled,
        )
    try:
        coordinator.notify_voice_turn(
            fish_id,
            session_id,
            message.sequence,
            message.id,
            voice_answer_message_id(turn_id),
            tracker,
            source_message=message,
            draft_lease=lease,
        )
    except BaseException:
        await _settle_voice_lease(lease)
        raise
    return AskResponse(
        message_id=message.id,
        session_id=message.session_id,
        sequence=message.sequence,
        status=message.status,
        created_at=message.created_at,
        answer_expected=True,
    )


@router.post(
    "/memories/recall",
    response_model=RecallResponse,
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def recall_facts(
    fish_id: UUID,
    body: RecallRequest,
    request: Request,
) -> RecallResponse | JSONResponse:
    if not body.query.strip():
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_content",
            "content 必须是非空字符串",
        )
    if len(body.query) > 6000:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "content_too_large",
            "content 不能超过 6000 个字符",
        )
    fish_store = cast(FishStore, request.app.state.fish_store)
    if not await fish_store.exists(fish_id):
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "fish 不存在",
        )

    service = cast(FactRecallService, request.app.state.recall_service)
    try:
        result = await service.recall(fish_id, body.query, 1)
    except RecallEmbeddingError:
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "embedding_unavailable",
            "embedding 服务暂时不可用",
        )
    except EmbeddingSpaceMismatchError as error:
        return error_response(
            status.HTTP_409_CONFLICT,
            "embedding_space_mismatch",
            str(error),
        )
    except Exception:
        logger.exception("Fact Recall failed for fish %s", fish_id)
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "recall_failed",
            "Recall 执行失败",
        )

    plan = result.trace.plan
    return RecallResponse(
        degraded=result.trace.degraded,
        facts=[
            RecallFactResponse(
                id=fact.id,
                source_message_id=fact.source_message_id,
                source_role=fact.source_role,
                text=fact.text,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                mentioned_at=fact.mentioned_at,
                entities=list(fact.entities),
                matched_by=list(fact.matched_by),
                semantic_score=fact.semantic_score,
                rank_score=fact.rank_score,
                invalidated_at=fact.invalidated_at,
            )
            for fact in result.facts
        ],
        trace=(
            None
            if not body.debug
            else RecallTraceResponse(
                depth=result.trace.depth,
                degraded=result.trace.degraded,
                plan=(
                    None
                    if plan is None
                    else RecallPlanResponse(
                        semantic_query=plan.semantic_query,
                        source_role=plan.source_role,
                        entity_hints=list(plan.entity_hints),
                        time=RecallTimePlanResponse(
                            start=plan.time.start,
                            end=plan.time.end,
                        ),
                    )
                ),
                plan_error=result.trace.plan_error,
                duration_ms=result.trace.duration_ms,
            )
        ),
    )


def _attachment_store(request: Request) -> MessageAttachmentStore:
    return cast(MessageAttachmentStore, request.app.state.message_attachments)


class AttachmentUpload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    kind: Literal["image", "text"]
    name: str = Field(max_length=200)
    mime_type: str = Field(validation_alias="mimeType", max_length=100)
    # Base64 image bytes, or the text of a text file.
    data: str | None = Field(default=None, max_length=9_000_000)
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    width: int | None = Field(default=None, ge=1, le=20_000)
    height: int | None = Field(default=None, ge=1, le=20_000)


@router.post("/attachments", status_code=status.HTTP_201_CREATED, response_model=None)
async def upload_attachment(
    fish_id: UUID, body: AttachmentUpload, request: Request
) -> dict[str, str] | JSONResponse:
    """Keeps a picture or text file until the message carrying it is sent."""
    store = _attachment_store(request)
    try:
        if body.kind == "image":
            try:
                data = base64.b64decode(body.data or "", validate=True)
            except ValueError:
                raise AttachmentError("图片数据无效") from None
            attachment_id = await store.add_image(
                fish_id,
                body.name,
                body.mime_type,
                data,
                width=body.width,
                height=body.height,
            )
        else:
            attachment_id = await store.add_text(
                fish_id, body.name, body.mime_type, body.text or ""
            )
    except AttachmentError as error:
        return error_response(
            status.HTTP_400_BAD_REQUEST, "invalid_attachment", str(error)
        )
    return {"attachmentId": str(attachment_id)}


@router.get("/attachments/{attachment_id}", response_model=None)
async def attachment_image(
    fish_id: UUID, attachment_id: UUID, request: Request
) -> Response | JSONResponse:
    image = await _attachment_store(request).image(fish_id, attachment_id)
    if image is None or image.data is None:
        return error_response(
            status.HTTP_404_NOT_FOUND, "attachment_not_found", "图片不存在"
        )
    return Response(content=image.data, media_type=image.mime_type)


@router.post(
    "/ask",
    response_model=AskResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def ask(
    fish_id: UUID,
    body: AskRequest,
    request: Request,
) -> AskResponse | JSONResponse:
    request_started_ns = perf_counter_ns()
    if not body.content.strip():
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_content",
            "content 必须是非空字符串",
        )
    if len(body.content) > 32_000:
        return error_response(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "content_too_large",
            "content 不能超过 32000 个字符",
        )

    reply_context = _freeze_reply_context(body.reply_context)
    reply_service = _agent_reply_service(request)
    if reply_context is not None and reply_service is None:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "agent_replies_unavailable",
            "Agent 精确回复服务未启用",
        )
    store = cast(MessagesStore, request.app.state.message_store)
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    attachments = _attachment_store(request) if body.attachment_ids else None
    if attachments is not None and not await attachments.waiting(
        fish_id, body.attachment_ids
    ):
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "attachment_expired",
            "附件已失效，请重新添加",
        )
    persist_started_ns = perf_counter_ns()
    handled = False
    inserted = True
    try:
        if reply_context is None:
            message = await store.insert_user_message(
                fish_id,
                body.content,
                session_id=body.session_id,
            )
        else:
            assert reply_service is not None and body.session_id is not None
            result = await reply_service.handle(
                fish_id,
                body.session_id,
                body.content,
                reply_context,
            )
            message = result.message
            handled = result.handled
            inserted = result.inserted
    except AgentReplyClassificationError:
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "reply_classification_failed",
            "Agent 回复分类失败，未消费 replyContext",
        )
    except AgentReplyConflictError:
        return error_response(
            status.HTTP_409_CONFLICT,
            "reply_context_conflict",
            "replyContext 已过期、已消费或不属于当前会话",
        )
    except MessageIdentityConflictError:
        return error_response(
            status.HTTP_409_CONFLICT,
            "reply_context_conflict",
            "回复 Message identity 与 replyContext 冲突",
        )
    except FishDeletedError:
        return error_response(
            status.HTTP_410_GONE,
            "fish_deleted",
            "fish 已删除",
        )
    except LegacyFishReadOnlyError:
        return error_response(
            status.HTTP_409_CONFLICT,
            "legacy_fish_read_only",
            "legacy fish 仅支持查看，不能继续对话",
        )
    except EmbeddingSpaceMismatchError as error:
        return error_response(
            status.HTTP_409_CONFLICT,
            "embedding_space_mismatch",
            str(error),
        )
    except SessionNotFoundError:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    await _bind_desktop_assistant_if_missing(request, fish_id)
    await _remember_input_mode(store, message, body.input_mode, inserted)
    if inserted and attachments is not None:
        # Before the reply starts, which reads them from its message.
        await attachments.attach(fish_id, message.id, body.attachment_ids)
    persist_finished_ns = perf_counter_ns()
    tracker = MessagePerformanceTracker(origin_ns=request_started_ns)
    tracker.record(
        "ask.persist_message",
        started_ns=persist_started_ns,
        finished_ns=persist_finished_ns,
    )
    if not handled and inserted:
        coordinator.notify_user_message(
            fish_id,
            message.session_id,
            message.sequence,
            message.id,
            tracker,
            source_message=message,
            **(
                {"playback_owner": "companion"}
                if body.playback_owner == "companion"
                else {}
            ),
        )
    return AskResponse(
        message_id=message.id,
        session_id=message.session_id,
        sequence=message.sequence,
        status=message.status,
        created_at=message.created_at,
        answer_expected=not handled,
    )


@router.get("/answer", response_model=None)
async def answer(
    fish_id: UUID,
    request: Request,
    after_sequence: int | None = Query(
        default=None,
        alias="afterSequence",
        ge=0,
    ),
    session_id: Annotated[UUID | None, Query(alias="sessionId")] = None,
) -> StreamingResponse | JSONResponse:
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    session = await _resolve_session(request, fish_id, session_id)
    if session is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "session_not_found",
            "session 不存在或不属于当前 fish",
        )
    runtime = coordinator.get_runtime(fish_id, session.id)

    async def stream() -> AsyncIterator[str]:
        async for event in iter_answer_events(
            runtime,
            after_sequence=after_sequence,
        ):
            yield format_sse_event(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/answer/progress", status_code=status.HTTP_204_NO_CONTENT)
async def update_progress(
    fish_id: UUID,
    body: ProgressRequest,
    request: Request,
) -> Response:
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    owner = getattr(coordinator, "answer_playback_owner", lambda *args: "web")(
        fish_id, body.message_id
    )
    if owner == "companion" and body.playback_owner != "companion":
        return error_response(
            409, "playback_owner_mismatch", "播放进度由 Companion 提交"
        )
    try:
        await coordinator.update_progress(
            fish_id,
            body.message_id,
            body.consumed_offset,
        )
    except ValueError as error:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_progress",
            str(error),
        )
    except LookupError:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "answer_not_found",
            "answer 不存在或不属于当前 fish",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/answer/interrupt", status_code=status.HTTP_204_NO_CONTENT)
async def interrupt_answer(
    fish_id: UUID,
    body: ProgressRequest,
    request: Request,
) -> Response:
    coordinator = cast(AnswerCoordinator, request.app.state.answer_coordinator)
    owner = getattr(coordinator, "answer_playback_owner", lambda *args: "web")(
        fish_id, body.message_id
    )
    hub = getattr(request.app.state, "pet_event_hub", None)
    terminal = (
        hub.terminal_answer_progress(fish_id, body.message_id)
        if hub is not None and body.playback_owner != "companion"
        else None
    )
    if terminal is not None:
        # Final playback state is emitted before the normal progress writer
        # finishes. Settle that device-confirmed cutoff before allowing Ask,
        # without waiting for an already-retired playback to interrupt again.
        offset = cast(int, terminal["consumedOffset"])
        try:
            if terminal["status"] == "interrupted":
                await coordinator.interrupt_answer(fish_id, body.message_id, offset)
            else:
                await coordinator.update_progress(fish_id, body.message_id, offset)
        except LookupError:
            return error_response(
                404, "answer_not_found", "answer 不存在或不属于当前 fish"
            )
        except ValueError as error:
            return error_response(400, "invalid_progress", str(error))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if owner == "companion" and body.playback_owner != "companion":
        try:
            if hub is None:
                raise TimeoutError
            await hub.request_answer_interrupt(fish_id, body.message_id)
        except TimeoutError:
            return error_response(
                503, "playback_unavailable", "Companion 尚未确认停止播报，请重试"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    try:
        await coordinator.interrupt_answer(
            fish_id,
            body.message_id,
            body.consumed_offset,
        )
    except ValueError as error:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_interrupt",
            str(error),
        )
    if hub is not None and body.playback_owner == "companion":
        hub.answer_interrupted(body.message_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
