import json
from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from fish_memory.models import MessageStatus, Role
from fish_memory.recall.models import RecallSourceRole
from fish_memory.tts.voice_config import VoiceKind


class ReplyContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["interaction", "run_result"]
    id: UUID


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    playback_owner: Literal["web", "companion"] = Field(
        default="web", validation_alias="playbackOwner"
    )

    content: str
    session_id: UUID | None = Field(default=None, validation_alias="sessionId")
    reply_context: ReplyContextRequest | None = Field(
        default=None,
        validation_alias="replyContext",
        serialization_alias="replyContext",
    )
    # Typed in a chat box, or spoken and transcribed; unknown for other callers.
    input_mode: Literal["typed", "spoken"] | None = Field(
        default=None, validation_alias="inputMode"
    )
    # Uploaded with POST /attachments; the content already names each one.
    attachment_ids: list[UUID] = Field(
        default_factory=list, validation_alias="attachmentIds", max_length=8
    )

    @model_validator(mode="after")
    def require_reply_session(self) -> Self:
        if self.reply_context is not None and self.session_id is None:
            raise ValueError("sessionId is required with replyContext")
        return self


class AskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: UUID = Field(serialization_alias="messageId")
    session_id: UUID = Field(serialization_alias="sessionId")
    sequence: int
    status: MessageStatus
    created_at: datetime = Field(serialization_alias="createdAt")
    answer_expected: bool = Field(
        default=True,
        serialization_alias="answerExpected",
    )


class SessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: UUID = Field(serialization_alias="sessionId")
    created_at: datetime = Field(serialization_alias="createdAt")
    last_message_at: datetime | None = Field(serialization_alias="lastMessageAt")


class SessionListResponse(BaseModel):
    items: list[SessionResponse]


class ProgressRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    playback_owner: Literal["web", "companion"] = Field(
        default="web", validation_alias="playbackOwner"
    )

    message_id: UUID = Field(validation_alias="messageId")
    consumed_offset: int = Field(validation_alias="consumedOffset", ge=0)


class VoiceDraftUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=32_000)
    reply_context: ReplyContextRequest | None = Field(
        default=None,
        validation_alias="replyContext",
        serialization_alias="replyContext",
    )
    speech_started_at: AwareDatetime | None = Field(
        default=None,
        validation_alias="speechStartedAt",
        serialization_alias="speechStartedAt",
    )
    frame_snapshot_at: AwareDatetime | None = Field(
        default=None,
        validation_alias="frameSnapshotAt",
        serialization_alias="frameSnapshotAt",
    )

    @model_validator(mode="after")
    def validate_frame_window(self) -> Self:
        if (self.speech_started_at is None) != (self.frame_snapshot_at is None):
            raise ValueError(
                "speechStartedAt and frameSnapshotAt must be provided together"
            )
        if (
            self.speech_started_at is not None
            and self.frame_snapshot_at is not None
            and self.speech_started_at > self.frame_snapshot_at
        ):
            raise ValueError("speechStartedAt must not follow frameSnapshotAt")
        return self


class VoiceDraftCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision: int = Field(ge=1)
    reason: Literal[
        "speech_resumed",
        "cancelled",
        "interrupted",
        "superseded",
        "expired",
        "rejected",
        "binding_changed",
        "shutdown",
    ]


class VoiceTurnCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision: int = Field(ge=1)
    promote_revision: int | None = Field(
        default=None,
        validation_alias="promoteRevision",
        serialization_alias="promoteRevision",
        ge=0,
    )
    content: str = Field(min_length=1, max_length=32_000)
    reply_context: ReplyContextRequest | None = Field(
        default=None,
        validation_alias="replyContext",
        serialization_alias="replyContext",
    )

    @model_validator(mode="after")
    def validate_revision_order(self) -> Self:
        if (
            self.promote_revision is not None
            and self.promote_revision > 0
            and self.promote_revision >= self.revision
        ):
            raise ValueError("promoteRevision must be earlier than revision")
        return self


class VoiceDraftStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    turn_id: UUID = Field(validation_alias="turnId", serialization_alias="turnId")
    revision: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=32_000)
    state: Literal["active", "cancelled", "committed", "expired"]


class RecallRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: str
    debug: bool = False


class RecallFactResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    source_message_id: UUID = Field(serialization_alias="sourceMessageId")
    source_role: Role = Field(serialization_alias="sourceRole")
    text: str
    occurred_start: datetime | None = Field(serialization_alias="occurredStart")
    occurred_end: datetime | None = Field(serialization_alias="occurredEnd")
    mentioned_at: datetime = Field(serialization_alias="mentionedAt")
    entities: list[str]
    matched_by: list[str] = Field(serialization_alias="matchedBy")
    semantic_score: float | None = Field(serialization_alias="semanticScore")
    rank_score: float | None = Field(serialization_alias="rankScore")
    invalidated_at: datetime | None = Field(serialization_alias="invalidatedAt")


class RecallTimePlanResponse(BaseModel):
    start: datetime | None
    end: datetime | None


class RecallPlanResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    semantic_query: str = Field(serialization_alias="semanticQuery")
    source_role: RecallSourceRole = Field(serialization_alias="sourceRole")
    entity_hints: list[str] = Field(serialization_alias="entityHints")
    time: RecallTimePlanResponse


class RecallTraceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    depth: int
    degraded: bool
    plan: RecallPlanResponse | None
    plan_error: str | None = Field(serialization_alias="planError")
    duration_ms: dict[str, float] = Field(serialization_alias="durationMs")


class RecallResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    degraded: bool
    facts: list[RecallFactResponse]
    trace: RecallTraceResponse | None = None


class FishSummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fish_id: UUID = Field(serialization_alias="fishId")
    last_message: str = Field(serialization_alias="lastMessage")
    last_active_at: datetime | None = Field(serialization_alias="lastActiveAt")
    message_count: int = Field(serialization_alias="messageCount")
    is_assistant: bool = Field(serialization_alias="isAssistant")
    name: str = ""
    role: str = ""
    evaluation: bool = False


class FishListResponse(BaseModel):
    items: list[FishSummaryResponse]


class AssistantFishResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fish_id: UUID | None = Field(serialization_alias="fishId")


class SoulProfileRequest(BaseModel):
    role: str | None = Field(default=None, max_length=200)
    setting: str | None = Field(default=None, max_length=200)
    name: str | None = Field(default=None, max_length=64)
    background: str | None = Field(default=None, max_length=1000)


class FishCreateRequest(SoulProfileRequest):
    pass


class FishCreateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fish_id: UUID = Field(serialization_alias="fishId")


class SoulUpdateRequest(SoulProfileRequest):
    pass


class SoulProfileResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str
    setting: str
    name: str
    background: str = ""


class WakeWordsRequest(BaseModel):
    words: list[str] = Field(min_length=1, max_length=4)

    @field_validator("words")
    @classmethod
    def clean_words(cls, value: list[str]) -> list[str]:
        words = [word.strip() for word in value]
        if any(not word or len(word) > 8 or "\n" in word for word in words):
            raise ValueError("唤醒词要 1 到 8 个字")
        if len(set(words)) != len(words):
            raise ValueError("唤醒词不能重复")
        return words


class WakeWordsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    words: list[str]
    # False while the Fish still uses the app-wide wake word.
    customized: bool


class MemoryForgetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    item_id: int | None = Field(default=None, alias="itemId", ge=1)
    fact_id: UUID | None = Field(default=None, alias="factId")

    @model_validator(mode="after")
    def exactly_one(self) -> "MemoryForgetRequest":
        if (self.item_id is None) == (self.fact_id is None):
            raise ValueError("itemId 和 factId 只能给一个")
        return self


class MemoryForgetResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_count: int = Field(serialization_alias="factCount")
    item_count: int = Field(serialization_alias="itemCount")


class FishVoiceConfigRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    voice: str = Field(min_length=1, max_length=160)
    kind: VoiceKind


class FishVoiceConfigResponse(FishVoiceConfigRequest):
    pass


class FishVoiceSelectionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str
    config: FishVoiceConfigResponse
    used_fallback: bool = Field(serialization_alias="usedFallback")


class SnapshotItemResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    text: str
    last_supported_at: datetime = Field(serialization_alias="lastSupportedAt")
    source_fact_count: int = Field(serialization_alias="sourceFactCount")


class SnapshotOverviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    snapshot_type: str = Field(serialization_alias="snapshotType")
    version: int
    updated_at: datetime = Field(serialization_alias="updatedAt")
    through_fact_sequence: int = Field(serialization_alias="throughFactSequence")
    items: list[SnapshotItemResponse]


class EntitySummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    entity_id: UUID = Field(serialization_alias="entityId")
    aliases: list[str]
    fact_count: int = Field(serialization_alias="factCount")


class EntityFactResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: UUID = Field(serialization_alias="factId")
    text: str
    occurred_start: datetime | None = Field(serialization_alias="occurredStart")
    occurred_end: datetime | None = Field(serialization_alias="occurredEnd")
    mentioned_at: datetime = Field(serialization_alias="mentionedAt")
    source_role: str = Field(serialization_alias="sourceRole")
    invalidated_at: datetime | None = Field(serialization_alias="invalidatedAt")


class EntityFactPageResponse(BaseModel):
    items: list[EntityFactResponse]
    has_more: bool = Field(serialization_alias="hasMore")


class FishManagementResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fish_id: UUID = Field(serialization_alias="fishId")
    is_assistant: bool = Field(serialization_alias="isAssistant")
    soul_profile: SoulProfileResponse = Field(serialization_alias="soulProfile")
    effective_soul: str = Field(serialization_alias="effectiveSoul")
    voice_config: FishVoiceConfigResponse | None = Field(
        serialization_alias="voiceConfig"
    )
    dynamic_snapshot: SnapshotOverviewResponse | None = Field(
        serialization_alias="dynamicSnapshot"
    )
    static_profile: SnapshotOverviewResponse | None = Field(
        serialization_alias="staticProfile"
    )
    entities: list[EntitySummaryResponse]
    created_at: datetime | None = Field(default=None, serialization_alias="createdAt")


class MessageOriginResponse(BaseModel):
    """Why TA spoke up on TA's own: something seen or heard (with the app, when
    known), another event, a plugin, a finished task, or TA's own idea."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["sense", "event", "plugin", "task", "self"]
    app_name: str | None = Field(default=None, serialization_alias="appName")


class MessageAttachmentResponse(BaseModel):
    id: UUID
    kind: Literal["image", "text"]
    name: str


class MessageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: UUID = Field(serialization_alias="messageId")
    sequence: int
    role: Role
    content: str
    status: MessageStatus
    consumed_offset: int | None = Field(serialization_alias="consumedOffset")
    created_at: datetime = Field(serialization_alias="createdAt")
    finished_at: datetime | None = Field(serialization_alias="finishedAt")
    origin: MessageOriginResponse | None = None
    # What TA had prepared but never said because the user cut in.
    unsaid: str | None = None
    # A user message typed in a chat box or spoken to the pet, when known.
    input_mode: Literal["typed", "spoken"] | None = Field(
        default=None, serialization_alias="inputMode"
    )
    # Pictures and files sent with a user message; a picture is read from
    # /v1/fishes/{fishId}/attachments/{id}.
    attachments: list[MessageAttachmentResponse] | None = None
    # Which outside service let this reply down ("chat:timeout"), if one did.
    failure: str | None = None
    # The conversation this message belongs to; the chat page marks where a
    # new one starts. Only the timeline and search fill it in.
    session_id: UUID | None = Field(default=None, serialization_alias="sessionId")


class MessageListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[MessageResponse]
    has_before: bool | None = Field(default=None, serialization_alias="hasBefore")
    has_after: bool | None = Field(default=None, serialization_alias="hasAfter")
    min_sequence: int | None = Field(default=None, serialization_alias="minSequence")
    max_sequence: int | None = Field(default=None, serialization_alias="maxSequence")


class PerformanceStepResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step_name: str = Field(serialization_alias="stepName")
    step_index: int = Field(serialization_alias="stepIndex")
    started_offset_us: int = Field(serialization_alias="startedOffsetUs")
    duration_us: int = Field(serialization_alias="durationUs")
    input_payload: JsonValue = Field(serialization_alias="inputPayload")
    output_payload: JsonValue = Field(serialization_alias="outputPayload")
    error: str | None


class HomeActionDebugResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    status: str
    action: JsonValue
    entity_id: str | None = Field(serialization_alias="entityId")
    service_domain: str | None = Field(serialization_alias="serviceDomain")
    service_name: str | None = Field(serialization_alias="serviceName")
    accepted: bool
    confirmed: bool
    result: JsonValue
    error: str | None
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")
    finished_at: datetime | None = Field(serialization_alias="finishedAt")


class PerformanceStepListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[PerformanceStepResponse]
    home_actions: list[HomeActionDebugResponse] = Field(
        default_factory=list,
        serialization_alias="homeActions",
    )


class TtsPerformanceRequest(BaseModel):
    metrics: dict[str, JsonValue]


class PetVoiceNativeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    endpoint_to_asr_final: float | None = Field(
        alias="endpointToAsrFinal", ge=0, allow_inf_nan=False
    )
    endpoint_to_ask: float | None = Field(
        alias="endpointToAsk", ge=0, allow_inf_nan=False
    )
    ask_to_answer_event: float | None = Field(
        alias="askToAnswerEvent", ge=0, allow_inf_nan=False
    )
    ask_to_first_content: float | None = Field(
        alias="askToFirstContent", ge=0, allow_inf_nan=False
    )
    first_content_to_tts_start: float | None = Field(
        alias="firstContentToTtsStart", ge=0, allow_inf_nan=False
    )
    tts_start_to_ready: float | None = Field(
        alias="ttsStartToReady", ge=0, allow_inf_nan=False
    )
    tts_ready_to_text_submitted: float | None = Field(
        alias="ttsReadyToTextSubmitted", ge=0, allow_inf_nan=False
    )
    tts_text_to_first_audio: float | None = Field(
        alias="ttsTextToFirstAudio", ge=0, allow_inf_nan=False
    )
    first_audio_to_pcm_queued: float | None = Field(
        alias="firstAudioToPcmQueued", ge=0, allow_inf_nan=False
    )
    pcm_queued_to_playback: float | None = Field(
        alias="pcmQueuedToPlayback", ge=0, allow_inf_nan=False
    )
    ask_to_playback: float | None = Field(
        alias="askToPlayback", ge=0, allow_inf_nan=False
    )
    endpoint_to_playback: float | None = Field(
        alias="endpointToPlayback", ge=0, allow_inf_nan=False
    )
    acoustic_end_to_first_content: float | None = Field(
        default=None,
        alias="acousticEndToFirstContent",
        ge=0,
        allow_inf_nan=False,
    )
    acoustic_end_to_playback: float | None = Field(
        default=None,
        alias="acousticEndToPlayback",
        ge=0,
        allow_inf_nan=False,
    )


class PetVoicePerformanceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: Literal["completed", "stopped", "failed"]
    user_message_id: UUID = Field(alias="userMessageId")
    answer_message_id: UUID = Field(alias="answerMessageId")
    provider: str | None
    model: str | None
    voice: str | None
    connection_reused: bool | None = Field(alias="connectionReused")
    native_ms: PetVoiceNativeMetrics = Field(alias="nativeMs")
    voice_draft: dict[str, JsonValue] | None = Field(
        default=None,
        alias="voiceDraft",
    )

    @field_validator("voice_draft")
    @classmethod
    def validate_voice_draft_json(
        cls,
        value: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("voiceDraft must contain finite JSON values") from error
        return value


class PetVoicePerformanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: PetVoicePerformanceMetrics


class BackgroundTaskProgressResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    through_sequence: int = Field(serialization_alias="throughSequence")
    target_sequence: int = Field(serialization_alias="targetSequence")
    pending: int
    last_error: str | None = Field(serialization_alias="lastError")
    updated_at: datetime | None = Field(serialization_alias="updatedAt")


class BackgroundProgressResponse(BaseModel):
    tasks: list[BackgroundTaskProgressResponse]
