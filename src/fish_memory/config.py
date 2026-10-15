from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.llm.configuration import DesktopModelRoutes, LanguageModelEndpoint

DEFAULT_SYSTEM_PROMPT = (
    "历史中标记为“被打断”的 assistant 内容，"
    "只代表你在被打断前实际说出的部分。\n"
    "请感知用户的打断和补充，始终优先回应最新消息。\n"
    "不要声称记得当前上下文中没有提供的信息。"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        loc_by_alias=False,
    )

    database_url: str = Field(
        default="pg0://fish-memory",
        min_length=1,
        validation_alias="FISH_MEMORY_DATABASE_URL",
    )
    database_data_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_DATABASE_DATA_DIR",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_API_KEY",
    )
    api_key_file: str = Field(
        default=".cache/fish-memory/api-key",
        min_length=1,
        validation_alias="FISH_MEMORY_API_KEY_FILE",
    )
    assistant_api_url: str = Field(
        default="http://127.0.0.1:8000",
        min_length=1,
        validation_alias="FISH_MEMORY_API_URL",
    )
    assistant_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        validation_alias="FISH_MEMORY_ASSISTANT_POLL_INTERVAL_SECONDS",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        validation_alias="FISH_MEMORY_SYSTEM_PROMPT",
    )
    llm_provider: Literal["openai"] = Field(
        validation_alias="FISH_MEMORY_LLM_PROVIDER",
    )
    llm_api_key: SecretStr = Field(
        validation_alias="FISH_MEMORY_LLM_API_KEY",
    )
    llm_model: str = Field(
        min_length=1,
        validation_alias="FISH_MEMORY_LLM_MODEL",
    )
    llm_base_url: str = Field(
        min_length=1,
        validation_alias="FISH_MEMORY_LLM_BASE_URL",
    )
    llm_log_retention_days: int = Field(
        default=7,
        ge=1,
        le=365,
        validation_alias="FISH_MEMORY_LLM_LOG_RETENTION_DAYS",
    )
    # Supplied by the desktop launcher, never required for .env/CLI use.
    desktop_model_routes: DesktopModelRoutes | None = Field(
        default=None, validation_alias="FISH_MEMORY_DESKTOP_MODEL_ROUTES"
    )

    def resolved_llm_routes(self) -> DesktopModelRoutes:
        if self.desktop_model_routes is not None:
            return self.desktop_model_routes
        endpoint = LanguageModelEndpoint(
            base_url=self.llm_base_url, model=self.llm_model, api_key=self.llm_api_key
        )
        return DesktopModelRoutes(chat=endpoint, reasoning=endpoint)

    answer_enable_search: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_ANSWER_ENABLE_SEARCH",
    )
    experimental_companion: ExperimentalConfiguration = Field(
        default_factory=ExperimentalConfiguration,
        validation_alias="FISH_MEMORY_EXPERIMENTAL_COMPANION",
    )
    codex_notifications_enabled: bool = Field(
        default=False, validation_alias="FISH_MEMORY_CODEX_NOTIFICATIONS_ENABLED"
    )
    codex_notifications_state_file: str = Field(
        default=".cache/fish-memory/codex-notifications.json",
        validation_alias="FISH_MEMORY_CODEX_NOTIFICATIONS_STATE_FILE",
    )
    proactive_task_cooldown_seconds: float = Field(
        default=600,
        ge=60,
        validation_alias="FISH_MEMORY_PROACTIVE_TASK_COOLDOWN_SECONDS",
    )
    proactive_task_starts_per_hour: int = Field(
        default=3,
        ge=1,
        le=12,
        validation_alias="FISH_MEMORY_PROACTIVE_TASK_STARTS_PER_HOUR",
    )
    proactive_task_timeout_seconds: float = Field(
        default=900,
        ge=30,
        le=3600,
        validation_alias="FISH_MEMORY_PROACTIVE_TASK_TIMEOUT_SECONDS",
    )
    answer_history_message_limit: int = Field(
        default=50,
        ge=1,
        validation_alias="FISH_MEMORY_ANSWER_HISTORY_MESSAGE_LIMIT",
    )
    home_assistant_enabled: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_HOME_ASSISTANT_ENABLED",
    )
    home_assistant_url: str = Field(
        default="http://127.0.0.1:8123",
        min_length=1,
        validation_alias="FISH_MEMORY_HOME_ASSISTANT_URL",
    )
    home_assistant_token: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_HOME_ASSISTANT_TOKEN",
    )
    home_router_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias="FISH_MEMORY_HOME_ROUTER_TIMEOUT_SECONDS",
    )
    home_router_concurrency: int = Field(
        default=8,
        ge=1,
        validation_alias="FISH_MEMORY_HOME_ROUTER_CONCURRENCY",
    )
    agent_tasks_enabled: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_AGENT_TASKS_ENABLED",
    )
    agent_url: str | None = Field(
        default=None,
        max_length=512,
        validation_alias="FISH_MEMORY_AGENT_URL",
    )
    agent_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_AGENT_API_KEY",
    )
    agent_credentials_file: str = Field(
        default=".cache/fish-memory/agent-credentials.json",
        min_length=1,
        validation_alias="FISH_MEMORY_AGENT_CREDENTIALS_FILE",
    )
    agent_run_concurrency: int = Field(
        default=4,
        ge=1,
        validation_alias="FISH_MEMORY_AGENT_RUN_CONCURRENCY",
    )
    agent_run_poll_interval_seconds: float = Field(
        default=2.0,
        gt=0,
        validation_alias="FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS",
    )
    agent_run_stale_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        validation_alias="FISH_MEMORY_AGENT_RUN_STALE_TIMEOUT_SECONDS",
    )
    agent_http_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        validation_alias="FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS",
    )
    agent_sse_read_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        validation_alias="FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS",
    )
    agent_reply_context_ttl_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="FISH_MEMORY_AGENT_REPLY_CONTEXT_TTL_SECONDS",
    )
    tts_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_TTS_API_KEY",
    )
    tts_provider: Literal["auto", "dashscope", "local"] = Field(
        default="auto",
        validation_alias="FISH_MEMORY_TTS_PROVIDER",
    )
    speech_facial_wait_for_ready: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY",
    )
    tts_websocket_url: str = Field(
        default="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        min_length=1,
        validation_alias="FISH_MEMORY_TTS_WEBSOCKET_URL",
    )
    tts_customization_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization",
        min_length=1,
        validation_alias="FISH_MEMORY_TTS_CUSTOMIZATION_URL",
    )
    tts_upload_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/uploads",
        min_length=1,
        validation_alias="FISH_MEMORY_TTS_UPLOAD_URL",
    )
    tts_local_model_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_TTS_LOCAL_MODEL_DIR",
    )
    tts_local_num_threads: int = Field(
        default=2,
        ge=1,
        validation_alias="FISH_MEMORY_TTS_LOCAL_NUM_THREADS",
    )
    desktop_pet_enabled: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_DESKTOP_PET_ENABLED",
    )
    desktop_mode: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_DESKTOP_MODE",
    )
    embedding_provider: Literal["local", "openai-compatible"] = Field(
        default="local",
        validation_alias="FISH_MEMORY_EMBEDDING_PROVIDER",
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_EMBEDDING_API_KEY",
    )
    embedding_base_url: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_EMBEDDING_BASE_URL",
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        min_length=1,
        validation_alias="FISH_MEMORY_EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=384,
        ge=1,
        le=2000,
        validation_alias="FISH_MEMORY_EMBEDDING_DIMENSIONS",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        validation_alias="FISH_MEMORY_EMBEDDING_BATCH_SIZE",
    )
    embedding_local_model_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR",
    )
    embedding_local_num_threads: int = Field(
        default=2,
        ge=1,
        validation_alias="FISH_MEMORY_EMBEDDING_LOCAL_NUM_THREADS",
    )
    embedding_local_max_tokens: int = Field(
        default=512,
        ge=1,
        validation_alias="FISH_MEMORY_EMBEDDING_LOCAL_MAX_TOKENS",
    )
    background_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        validation_alias="FISH_MEMORY_BACKGROUND_POLL_INTERVAL_SECONDS",
    )
    background_llm_concurrency: int = Field(
        default=1,
        ge=1,
        validation_alias="FISH_MEMORY_BACKGROUND_LLM_CONCURRENCY",
    )
    background_embedding_concurrency: int = Field(
        default=4,
        ge=1,
        validation_alias="FISH_MEMORY_BACKGROUND_EMBEDDING_CONCURRENCY",
    )
    fact_extraction_max_retries: int = Field(
        default=3,
        ge=1,
        validation_alias="FISH_MEMORY_FACT_EXTRACTION_MAX_RETRIES",
    )
    fact_llm_timeout_seconds: float = Field(
        default=120,
        gt=0,
        validation_alias="FISH_MEMORY_FACT_LLM_TIMEOUT_SECONDS",
    )
    fact_batch_size: int = Field(
        default=8,
        ge=1,
        validation_alias="FISH_MEMORY_FACT_BATCH_SIZE",
    )
    dynamic_snapshot_max_age_days: int = Field(
        default=7,
        ge=1,
        validation_alias="FISH_MEMORY_DYNAMIC_SNAPSHOT_MAX_AGE_DAYS",
    )
    snapshot_batch_size: int = Field(
        default=100,
        ge=1,
        validation_alias="FISH_MEMORY_SNAPSHOT_BATCH_SIZE",
    )
    history_token_limit: int = Field(
        default=20_000,
        ge=1,
        validation_alias="FISH_MEMORY_HISTORY_TOKEN_LIMIT",
    )
    history_token_retain: int = Field(
        default=10_000,
        ge=1,
        validation_alias="FISH_MEMORY_HISTORY_TOKEN_RETAIN",
    )
    recall_context_token_budget: int = Field(
        default=50_000,
        ge=1,
        validation_alias="FISH_MEMORY_RECALL_CONTEXT_TOKEN_BUDGET",
    )
    kws_model_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_KWS_MODEL_DIR",
    )
    wake_word: str = Field(
        default="小鱼",
        min_length=1,
        validation_alias="FISH_MEMORY_WAKE_WORD",
    )
    kws_chunk_size: Literal[8, 16] = Field(
        default=8,
        validation_alias="FISH_MEMORY_KWS_CHUNK_SIZE",
    )
    kws_num_threads: int = Field(
        default=1,
        ge=1,
        validation_alias="FISH_MEMORY_KWS_NUM_THREADS",
    )
    kws_score: float = Field(
        default=1.0,
        gt=0,
        validation_alias="FISH_MEMORY_KWS_SCORE",
    )
    kws_threshold: float = Field(
        default=0.25,
        ge=0,
        validation_alias="FISH_MEMORY_KWS_THRESHOLD",
    )
    asr_model: Literal["sensevoice"] = Field(
        default="sensevoice",
        validation_alias="FISH_MEMORY_ASR_MODEL",
    )
    asr_model_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_ASR_MODEL_DIR",
    )
    asr_num_threads: int = Field(
        default=2,
        ge=1,
        validation_alias="FISH_MEMORY_ASR_NUM_THREADS",
    )
    companion_audio_block_size: int = Field(
        default=160,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_AUDIO_BLOCK_SIZE",
    )
    companion_microphone_enabled: bool = Field(
        default=True,
        validation_alias="FISH_MEMORY_COMPANION_MICROPHONE_ENABLED",
    )
    companion_playback_enabled: bool = Field(
        default=True,
        validation_alias="FISH_MEMORY_COMPANION_PLAYBACK_ENABLED",
    )
    companion_audio_backend: Literal["sounddevice", "livekit"] = Field(
        default="sounddevice",
        validation_alias="FISH_MEMORY_COMPANION_AUDIO_BACKEND",
    )
    livekit_audio_helper: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_LIVEKIT_AUDIO_HELPER",
    )
    owner_voice_enabled: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_OWNER_VOICE_ENABLED",
    )
    owner_voice_dir: str = Field(
        default=".cache/fish-memory/owner-voice",
        min_length=1,
        validation_alias="FISH_MEMORY_OWNER_VOICE_DIR",
    )
    owner_voice_threshold: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        validation_alias="FISH_MEMORY_OWNER_VOICE_THRESHOLD",
    )
    environment_history_speaker_enabled: bool = Field(
        default=True,
        validation_alias="FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_ENABLED",
    )
    environment_history_speaker_threshold: float = Field(
        default=0.65,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        validation_alias="FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_THRESHOLD",
    )
    environment_history_speaker_margin: float = Field(
        default=0.10,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        validation_alias="FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_MARGIN",
    )
    environment_history_speaker_max_speakers: int = Field(
        default=16,
        ge=2,
        le=64,
        validation_alias="FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_MAX_SPEAKERS",
    )
    companion_input_device: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_COMPANION_INPUT_DEVICE",
    )
    companion_output_device: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_COMPANION_OUTPUT_DEVICE",
    )
    companion_aec_delay_ms: int = Field(
        default=50,
        ge=0,
        validation_alias="FISH_MEMORY_COMPANION_AEC_DELAY_MS",
    )
    companion_aec_delay_mode: Literal["fixed", "auto"] = Field(
        default="auto",
        validation_alias="FISH_MEMORY_COMPANION_AEC_DELAY_MODE",
    )
    companion_vad_mode: int = Field(
        default=2,
        ge=0,
        le=3,
        validation_alias="FISH_MEMORY_COMPANION_VAD_MODE",
    )
    companion_vad_start_frames: int = Field(
        default=1,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_VAD_START_FRAMES",
    )
    companion_vad_model_dir: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_COMPANION_VAD_MODEL_DIR",
    )
    companion_vad_threshold: float = Field(
        default=0.25,
        ge=0,
        le=1,
        validation_alias="FISH_MEMORY_COMPANION_VAD_THRESHOLD",
    )
    companion_vad_min_silence_ms: int = Field(
        default=350,
        ge=10,
        validation_alias="FISH_MEMORY_COMPANION_VAD_MIN_SILENCE_MS",
    )
    companion_vad_min_speech_ms: int = Field(
        default=80,
        ge=10,
        validation_alias="FISH_MEMORY_COMPANION_VAD_MIN_SPEECH_MS",
    )
    companion_vad_window_size: int = Field(
        default=512,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_VAD_WINDOW_SIZE",
    )
    companion_vad_max_speech_seconds: float = Field(
        default=20.0,
        gt=0,
        validation_alias="FISH_MEMORY_COMPANION_VAD_MAX_SPEECH_SECONDS",
    )
    companion_vad_num_threads: int = Field(
        default=1,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_VAD_NUM_THREADS",
    )
    companion_vad_end_silence_ms: int = Field(
        default=200,
        ge=10,
        validation_alias="FISH_MEMORY_COMPANION_VAD_END_SILENCE_MS",
    )
    companion_utterance_commit_delay_ms: int = Field(
        default=0,
        ge=0,
        validation_alias="FISH_MEMORY_COMPANION_UTTERANCE_COMMIT_DELAY_MS",
    )
    companion_smart_turn_enabled: bool = Field(
        default=True,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_ENABLED",
    )
    companion_smart_turn_model_path: str | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_MODEL_PATH",
    )
    companion_smart_turn_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_THRESHOLD",
    )
    companion_smart_turn_reopen_ms: int = Field(
        default=800,
        ge=0,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_REOPEN_MS",
    )
    companion_smart_turn_timeout_ms: int = Field(
        default=500,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_TIMEOUT_MS",
    )
    companion_smart_turn_num_threads: int = Field(
        default=1,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_SMART_TURN_NUM_THREADS",
    )
    companion_voice_draft_enabled: bool = Field(
        default=True,
        validation_alias="FISH_MEMORY_COMPANION_VOICE_DRAFT_ENABLED",
    )
    companion_voice_draft_candidate_silence_ms: int = Field(
        default=100,
        ge=0,
        description=(
            "Continuous VAD non-speech before authorizing an LLM draft. "
            "ASR prepares from the first quiet frame; 0 removes the extra wait."
        ),
        validation_alias="FISH_MEMORY_COMPANION_VOICE_DRAFT_CANDIDATE_SILENCE_MS",
    )
    companion_voice_draft_ttl_ms: int = Field(
        default=2000,
        ge=2000,
        le=2000,
        validation_alias="FISH_MEMORY_COMPANION_VOICE_DRAFT_TTL_MS",
    )
    companion_voice_draft_max_concurrency: int = Field(
        default=2,
        ge=1,
        le=2,
        validation_alias="FISH_MEMORY_COMPANION_VOICE_DRAFT_MAX_CONCURRENCY",
    )
    companion_voice_draft_tts_max_buffer_ms: int = Field(
        default=3000,
        ge=0,
        le=3000,
        validation_alias="FISH_MEMORY_COMPANION_VOICE_DRAFT_TTS_MAX_BUFFER_MS",
    )
    companion_early_interrupt_snapshot_ms: int = Field(
        default=300,
        ge=0,
        validation_alias="FISH_MEMORY_COMPANION_EARLY_INTERRUPT_SNAPSHOT_MS",
    )
    companion_early_interrupt_min_chars: int = Field(
        default=2,
        ge=1,
        validation_alias="FISH_MEMORY_COMPANION_EARLY_INTERRUPT_MIN_CHARS",
    )
    companion_single_char_min_speech_ms: int = Field(
        default=200,
        ge=0,
        validation_alias="FISH_MEMORY_COMPANION_SINGLE_CHAR_MIN_SPEECH_MS",
    )
    audio_diagnostics_enabled: bool = Field(
        default=False,
        validation_alias="FISH_MEMORY_AUDIO_DIAGNOSTICS_ENABLED",
    )
    audio_diagnostics_dir: str = Field(
        default=".cache/fish-memory/audio-diagnostics",
        min_length=1,
        validation_alias="FISH_MEMORY_AUDIO_DIAGNOSTICS_DIR",
    )
    audio_diagnostics_retention_days: int = Field(
        default=7,
        ge=1,
        validation_alias="FISH_MEMORY_AUDIO_DIAGNOSTICS_RETENTION_DAYS",
    )
    audio_diagnostics_pre_seconds: float = Field(
        default=5.0,
        ge=0,
        validation_alias="FISH_MEMORY_AUDIO_DIAGNOSTICS_PRE_SECONDS",
    )
    audio_diagnostics_post_seconds: float = Field(
        default=5.0,
        ge=0,
        validation_alias="FISH_MEMORY_AUDIO_DIAGNOSTICS_POST_SECONDS",
    )

    @field_validator("asr_model", mode="before")
    @classmethod
    def normalize_legacy_asr_model(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {
            "paraformer",
            "zipformer",
        }:
            return "sensevoice"
        return value

    @field_validator("agent_url", mode="before")
    @classmethod
    def normalize_optional_agent_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("agent_api_key", mode="before")
    @classmethod
    def normalize_optional_agent_api_key(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            normalized = value.get_secret_value().strip()
            return SecretStr(normalized) if normalized else None
        if isinstance(value, str):
            normalized = value.strip()
            return SecretStr(normalized) if normalized else None
        return value

    @field_validator("agent_credentials_file", mode="before")
    @classmethod
    def normalize_agent_credentials_file(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if (
            self.environment_history_speaker_margin
            >= self.environment_history_speaker_threshold
        ):
            raise ValueError(
                "speaker matching margin must be smaller than its threshold"
            )
        if not self.llm_api_key.get_secret_value().strip():
            raise ValueError("FISH_MEMORY_LLM_API_KEY is required and cannot be blank")
        if "history_token_retain" not in self.model_fields_set:
            self.history_token_retain = max(1, self.history_token_limit // 2)
        if self.history_token_retain >= self.history_token_limit:
            raise ValueError(
                "history_token_retain must be smaller than history_token_limit"
            )
        if self.tts_provider == "dashscope" and self.tts_api_key is None:
            raise ValueError(
                "FISH_MEMORY_TTS_API_KEY is required when "
                "FISH_MEMORY_TTS_PROVIDER=dashscope"
            )
        if self.embedding_provider == "openai-compatible":
            if self.embedding_api_key is None:
                raise ValueError(
                    "FISH_MEMORY_EMBEDDING_API_KEY is required when "
                    "FISH_MEMORY_EMBEDDING_PROVIDER=openai-compatible"
                )
            if not self.embedding_base_url:
                raise ValueError(
                    "FISH_MEMORY_EMBEDDING_BASE_URL is required when "
                    "FISH_MEMORY_EMBEDDING_PROVIDER=openai-compatible"
                )
        if self.home_assistant_enabled and self.home_assistant_token is None:
            raise ValueError(
                "FISH_MEMORY_HOME_ASSISTANT_TOKEN is required when "
                "FISH_MEMORY_HOME_ASSISTANT_ENABLED=true"
            )
        return self
