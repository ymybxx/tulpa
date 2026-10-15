from collections.abc import Iterator
from typing import Any, cast

import pytest
from pydantic import ValidationError

from fish_memory.config import DEFAULT_SYSTEM_PROMPT, Settings

LLM_ENV = {
    "FISH_MEMORY_LLM_PROVIDER": "openai",
    "FISH_MEMORY_LLM_API_KEY": "test-key",
    "FISH_MEMORY_LLM_MODEL": "qwen-plus",
    "FISH_MEMORY_LLM_BASE_URL": "https://example.invalid/v1",
}
ANSWER_ENV = {
    "FISH_MEMORY_ANSWER_ENABLE_SEARCH": "false",
}
BACKGROUND_ENV = {
    "FISH_MEMORY_ANSWER_HISTORY_MESSAGE_LIMIT": "50",
    "FISH_MEMORY_BACKGROUND_POLL_INTERVAL_SECONDS": "0.25",
    "FISH_MEMORY_BACKGROUND_LLM_CONCURRENCY": "4",
    "FISH_MEMORY_BACKGROUND_EMBEDDING_CONCURRENCY": "6",
    "FISH_MEMORY_FACT_EXTRACTION_MAX_RETRIES": "5",
    "FISH_MEMORY_FACT_LLM_TIMEOUT_SECONDS": "120",
    "FISH_MEMORY_FACT_BATCH_SIZE": "12",
    "FISH_MEMORY_DYNAMIC_SNAPSHOT_MAX_AGE_DAYS": "9",
    "FISH_MEMORY_SNAPSHOT_BATCH_SIZE": "50",
    "FISH_MEMORY_HISTORY_TOKEN_LIMIT": "160000",
    "FISH_MEMORY_HISTORY_TOKEN_RETAIN": "110000",
    "FISH_MEMORY_API_URL": "http://127.0.0.1:8100",
    "FISH_MEMORY_ASSISTANT_POLL_INTERVAL_SECONDS": "2.5",
    "FISH_MEMORY_ASR_NUM_THREADS": "3",
    "FISH_MEMORY_COMPANION_AUDIO_BLOCK_SIZE": "640",
    "FISH_MEMORY_COMPANION_AUDIO_BACKEND": "livekit",
    "FISH_MEMORY_COMPANION_MICROPHONE_ENABLED": "false",
    "FISH_MEMORY_COMPANION_PLAYBACK_ENABLED": "true",
    "FISH_MEMORY_LIVEKIT_AUDIO_HELPER": "/tmp/fish-livekit-audio-helper",
    "FISH_MEMORY_COMPANION_AEC_DELAY_MS": "70",
    "FISH_MEMORY_COMPANION_AEC_DELAY_MODE": "auto",
    "FISH_MEMORY_COMPANION_VAD_MODE": "3",
    "FISH_MEMORY_COMPANION_VAD_START_FRAMES": "3",
    "FISH_MEMORY_COMPANION_VAD_MODEL_DIR": "/tmp/fish-vad",
    "FISH_MEMORY_COMPANION_VAD_THRESHOLD": "0.4",
    "FISH_MEMORY_COMPANION_VAD_MIN_SILENCE_MS": "420",
    "FISH_MEMORY_COMPANION_VAD_MIN_SPEECH_MS": "90",
    "FISH_MEMORY_COMPANION_VAD_WINDOW_SIZE": "512",
    "FISH_MEMORY_COMPANION_VAD_MAX_SPEECH_SECONDS": "30",
    "FISH_MEMORY_COMPANION_VAD_NUM_THREADS": "2",
    "FISH_MEMORY_COMPANION_VAD_END_SILENCE_MS": "250",
    "FISH_MEMORY_COMPANION_UTTERANCE_COMMIT_DELAY_MS": "180",
    "FISH_MEMORY_COMPANION_SMART_TURN_ENABLED": "false",
    "FISH_MEMORY_COMPANION_SMART_TURN_MODEL_PATH": "/tmp/smart-turn.onnx",
    "FISH_MEMORY_COMPANION_SMART_TURN_THRESHOLD": "0.65",
    "FISH_MEMORY_COMPANION_SMART_TURN_REOPEN_MS": "900",
    "FISH_MEMORY_COMPANION_SMART_TURN_TIMEOUT_MS": "650",
    "FISH_MEMORY_COMPANION_SMART_TURN_NUM_THREADS": "2",
    "FISH_MEMORY_COMPANION_EARLY_INTERRUPT_SNAPSHOT_MS": "240",
    "FISH_MEMORY_COMPANION_EARLY_INTERRUPT_MIN_CHARS": "3",
    "FISH_MEMORY_COMPANION_SINGLE_CHAR_MIN_SPEECH_MS": "260",
    "FISH_MEMORY_AUDIO_DIAGNOSTICS_ENABLED": "true",
    "FISH_MEMORY_AUDIO_DIAGNOSTICS_DIR": "/tmp/fish-audio",
    "FISH_MEMORY_AUDIO_DIAGNOSTICS_RETENTION_DAYS": "14",
    "FISH_MEMORY_AUDIO_DIAGNOSTICS_PRE_SECONDS": "4.5",
    "FISH_MEMORY_AUDIO_DIAGNOSTICS_POST_SECONDS": "6.5",
}
EMBEDDING_ENV = {
    "FISH_MEMORY_EMBEDDING_PROVIDER": "openai-compatible",
    "FISH_MEMORY_EMBEDDING_API_KEY": "embedding-only-key",
    "FISH_MEMORY_EMBEDDING_BASE_URL": "https://embedding.example.invalid/v1",
    "FISH_MEMORY_EMBEDDING_MODEL": "text-embedding-v4",
    "FISH_MEMORY_EMBEDDING_DIMENSIONS": "1024",
    "FISH_MEMORY_EMBEDDING_BATCH_SIZE": "16",
}
LEGACY_LLM_ENV = {
    name.replace("FISH_MEMORY_", "HINDSIGHT_API_"): value
    for name, value in LLM_ENV.items()
}


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    names = {
        "FISH_MEMORY_DATABASE_URL",
        "FISH_MEMORY_LLM_LOG_RETENTION_DAYS",
        "FISH_MEMORY_SYSTEM_PROMPT",
        *LLM_ENV,
        *ANSWER_ENV,
        *BACKGROUND_ENV,
        *EMBEDDING_ENV,
        *LEGACY_LLM_ENV,
        "FISH_MEMORY_API_URL",
        "FISH_MEMORY_ASSISTANT_POLL_INTERVAL_SECONDS",
        "FISH_MEMORY_AUDIO_DIAGNOSTICS_ENABLED",
        "FISH_MEMORY_AUDIO_DIAGNOSTICS_DIR",
        "FISH_MEMORY_AUDIO_DIAGNOSTICS_RETENTION_DAYS",
        "FISH_MEMORY_AUDIO_DIAGNOSTICS_PRE_SECONDS",
        "FISH_MEMORY_AUDIO_DIAGNOSTICS_POST_SECONDS",
        "FISH_MEMORY_TTS_PROVIDER",
        "FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY",
        "FISH_MEMORY_TTS_API_KEY",
        "FISH_MEMORY_TTS_LOCAL_MODEL_DIR",
        "FISH_MEMORY_TTS_LOCAL_NUM_THREADS",
        "FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR",
        "FISH_MEMORY_EMBEDDING_LOCAL_NUM_THREADS",
        "FISH_MEMORY_EMBEDDING_LOCAL_MAX_TOKENS",
        "FISH_MEMORY_HOME_ASSISTANT_ENABLED",
        "FISH_MEMORY_HOME_ASSISTANT_URL",
        "FISH_MEMORY_HOME_ASSISTANT_TOKEN",
        "FISH_MEMORY_HOME_ROUTER_TIMEOUT_SECONDS",
        "FISH_MEMORY_HOME_ROUTER_CONCURRENCY",
        "FISH_MEMORY_DESKTOP_PET_ENABLED",
        "FISH_MEMORY_AGENT_TASKS_ENABLED",
        "FISH_MEMORY_AGENT_URL",
        "FISH_MEMORY_AGENT_API_KEY",
        "FISH_MEMORY_AGENT_CREDENTIALS_FILE",
        "FISH_MEMORY_AGENT_DEFAULT_TARGET_ID",
        "FISH_MEMORY_AGENT_CREDENTIALS",
        "FISH_MEMORY_AGENT_RUN_CONCURRENCY",
        "FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS",
        "FISH_MEMORY_AGENT_RUN_STALE_TIMEOUT_SECONDS",
        "FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS",
        "FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS",
        "FISH_MEMORY_AGENT_REPLY_CONTEXT_TTL_SECONDS",
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)
    yield


def set_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in LLM_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("microphone", [False, True])
@pytest.mark.parametrize("playback", [False, True])
def test_desktop_audio_switches_are_loaded_independently_at_startup(
    monkeypatch: pytest.MonkeyPatch, microphone: bool, playback: bool
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(
        "FISH_MEMORY_COMPANION_MICROPHONE_ENABLED", str(microphone).lower()
    )
    monkeypatch.setenv("FISH_MEMORY_COMPANION_PLAYBACK_ENABLED", str(playback).lower())
    settings = settings_without_env_file()
    assert settings.companion_microphone_enabled is microphone
    assert settings.companion_playback_enabled is playback


@pytest.mark.parametrize(
    ("threshold", "margin", "capacity"),
    [
        ("nan", "0.1", "16"),
        ("0.65", "0.65", "16"),
        ("0.65", "0", "16"),
        ("0.65", "0.1", "100"),
    ],
)
def test_rejects_invalid_system_audio_speaker_matching(
    monkeypatch: pytest.MonkeyPatch, threshold: str, margin: str, capacity: str
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_THRESHOLD", threshold)
    monkeypatch.setenv("FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_MARGIN", margin)
    monkeypatch.setenv("FISH_MEMORY_ENVIRONMENT_HISTORY_SPEAKER_MAX_SPEAKERS", capacity)
    with pytest.raises(ValidationError):
        settings_without_env_file()


def settings_without_env_file() -> Settings:
    settings_type = cast(Any, Settings)
    return settings_type(_env_file=None)


def test_uses_embedded_database_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.database_url == "pg0://fish-memory"


@pytest.mark.parametrize("days", [None, "1", "7", "30", "365"])
def test_llm_log_retention_default_and_environment(monkeypatch, days):
    set_llm_environment(monkeypatch)
    if days is not None:
        monkeypatch.setenv("FISH_MEMORY_LLM_LOG_RETENTION_DAYS", days)
    assert settings_without_env_file().llm_log_retention_days == (
        7 if days is None else int(days)
    )


@pytest.mark.parametrize("days", ["0", "-1", "366", "1.5", "true"])
def test_llm_log_retention_rejects_invalid_environment(monkeypatch, days):
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_LLM_LOG_RETENTION_DAYS", days)
    with pytest.raises(ValidationError):
        settings_without_env_file()


@pytest.mark.parametrize("configured", [None, "true", "false"])
def test_speech_facial_wait_defaults_off_and_reads_environment(
    monkeypatch: pytest.MonkeyPatch, configured: str | None
) -> None:
    set_llm_environment(monkeypatch)
    if configured is not None:
        monkeypatch.setenv("FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY", configured)
    assert settings_without_env_file().speech_facial_wait_for_ready is (
        configured == "true"
    )


def test_tts_defaults_to_automatic_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.tts_provider == "auto"


def test_companion_audio_backend_defaults_to_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.companion_audio_backend == "sounddevice"
    assert settings.livekit_audio_helper is None
    assert settings.tts_api_key is None
    assert settings.tts_local_model_dir is None
    assert settings.tts_local_num_threads == 2


def test_home_assistant_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.home_assistant_enabled is False
    assert settings.home_assistant_url == "http://127.0.0.1:8123"
    assert settings.home_assistant_token is None
    assert settings.home_router_timeout_seconds == 5.0
    assert settings.home_router_concurrency == 8


def test_desktop_pet_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.desktop_pet_enabled is False


def test_agent_settings_default_to_an_empty_optional_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.agent_tasks_enabled is False
    assert settings.agent_url is None
    assert settings.agent_api_key is None
    assert settings.agent_credentials_file == (
        ".cache/fish-memory/agent-credentials.json"
    )
    assert not hasattr(settings, "agent_default_target_id")
    assert not hasattr(settings, "agent_credentials")
    assert settings.agent_run_concurrency == 4
    assert settings.agent_run_poll_interval_seconds == 2.0
    assert settings.agent_run_stale_timeout_seconds == 300.0
    assert settings.agent_http_timeout_seconds == 15.0
    assert settings.agent_sse_read_timeout_seconds == 45.0
    assert settings.agent_reply_context_ttl_seconds == 30.0


def test_agent_global_endpoint_uses_one_url_and_one_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(
        "FISH_MEMORY_AGENT_URL",
        "  http://127.0.0.1:8642/p/coder/  ",
    )
    monkeypatch.setenv("FISH_MEMORY_AGENT_API_KEY", "  private-agent-key  ")
    monkeypatch.setenv(
        "FISH_MEMORY_AGENT_CREDENTIALS_FILE",
        "  /tmp/fish-memory-agent-credentials.json  ",
    )
    monkeypatch.setenv("FISH_MEMORY_AGENT_RUN_STALE_TIMEOUT_SECONDS", "412.5")

    settings = settings_without_env_file()

    assert settings.agent_url == "http://127.0.0.1:8642/p/coder/"
    assert settings.agent_api_key is not None
    assert settings.agent_api_key.get_secret_value() == "private-agent-key"
    assert settings.agent_credentials_file == "/tmp/fish-memory-agent-credentials.json"
    assert settings.agent_run_stale_timeout_seconds == 412.5


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("FISH_MEMORY_AGENT_RUN_CONCURRENCY", "0"),
        ("FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS", "0"),
        ("FISH_MEMORY_AGENT_RUN_STALE_TIMEOUT_SECONDS", "0"),
        ("FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS", "0"),
        ("FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS", "0"),
        ("FISH_MEMORY_AGENT_REPLY_CONTEXT_TTL_SECONDS", "0"),
    ],
)
def test_agent_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        settings_without_env_file()


def test_reads_desktop_pet_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_DESKTOP_PET_ENABLED", "true")

    settings = settings_without_env_file()

    assert settings.desktop_pet_enabled is True


def test_voice_draft_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    settings = settings_without_env_file()
    assert settings.companion_voice_draft_enabled is True
    assert settings.companion_voice_draft_candidate_silence_ms == 100
    assert settings.companion_voice_draft_ttl_ms == 2000
    assert settings.companion_voice_draft_max_concurrency == 2
    assert settings.companion_voice_draft_tts_max_buffer_ms == 3000


def test_voice_draft_can_be_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_VOICE_DRAFT_ENABLED", "false")

    assert settings_without_env_file().companion_voice_draft_enabled is False


def test_voice_draft_wait_can_be_removed_without_disabling_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_VOICE_DRAFT_CANDIDATE_SILENCE_MS", "0")
    settings = settings_without_env_file()

    assert settings.companion_voice_draft_candidate_silence_ms == 0
    assert settings.companion_voice_draft_enabled is True


def test_voice_draft_tts_buffer_rejects_values_over_three_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_VOICE_DRAFT_TTS_MAX_BUFFER_MS", "3001")

    with pytest.raises(ValidationError):
        settings_without_env_file()


@pytest.mark.parametrize("ttl_ms", ["1999", "2001"])
def test_voice_draft_ttl_requires_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
    ttl_ms: str,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_VOICE_DRAFT_TTL_MS", ttl_ms)

    with pytest.raises(ValidationError):
        settings_without_env_file()


@pytest.mark.parametrize("concurrency", ["0", "3"])
def test_voice_draft_concurrency_is_bounded_to_two(
    monkeypatch: pytest.MonkeyPatch,
    concurrency: str,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(
        "FISH_MEMORY_COMPANION_VOICE_DRAFT_MAX_CONCURRENCY",
        concurrency,
    )

    with pytest.raises(ValidationError):
        settings_without_env_file()


def test_voice_draft_concurrency_accepts_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(
        "FISH_MEMORY_COMPANION_VOICE_DRAFT_MAX_CONCURRENCY",
        "1",
    )
    monkeypatch.setenv("FISH_MEMORY_COMPANION_VOICE_DRAFT_TTL_MS", "2000")

    settings = settings_without_env_file()
    assert settings.companion_voice_draft_max_concurrency == 1
    assert settings.companion_voice_draft_ttl_ms == 2000


def test_home_assistant_requires_token_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_HOME_ASSISTANT_ENABLED", "true")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "FISH_MEMORY_HOME_ASSISTANT_TOKEN" in str(exc_info.value)


def test_reads_home_assistant_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_HOME_ASSISTANT_ENABLED", "true")
    monkeypatch.setenv("FISH_MEMORY_HOME_ASSISTANT_URL", "http://ha.local:8123/")
    monkeypatch.setenv("FISH_MEMORY_HOME_ASSISTANT_TOKEN", "ha-token")
    monkeypatch.setenv("FISH_MEMORY_HOME_ROUTER_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("FISH_MEMORY_HOME_ROUTER_CONCURRENCY", "12")

    settings = settings_without_env_file()

    assert settings.home_assistant_enabled is True
    assert settings.home_assistant_url == "http://ha.local:8123/"
    assert settings.home_assistant_token is not None
    assert settings.home_assistant_token.get_secret_value() == "ha-token"
    assert settings.home_router_timeout_seconds == 3.5
    assert settings.home_router_concurrency == 12


def test_reads_independent_tts_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_TTS_PROVIDER", "dashscope")
    monkeypatch.setenv("FISH_MEMORY_TTS_API_KEY", "tts-only-key")
    monkeypatch.setenv("FISH_MEMORY_TTS_LOCAL_MODEL_DIR", "/tmp/fish-tts")
    monkeypatch.setenv("FISH_MEMORY_TTS_LOCAL_NUM_THREADS", "4")

    settings = settings_without_env_file()

    assert settings.tts_provider == "dashscope"
    assert settings.tts_api_key is not None
    assert settings.tts_api_key.get_secret_value() == "tts-only-key"
    assert settings.tts_local_model_dir == "/tmp/fish-tts"
    assert settings.tts_local_num_threads == 4


def test_rejects_dashscope_tts_without_independent_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_TTS_PROVIDER", "dashscope")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "FISH_MEMORY_TTS_API_KEY" in str(exc_info.value)


def test_reads_bailian_openai_compatible_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FISH_MEMORY_DATABASE_URL",
        "postgresql://fish:secret@localhost/fish_memory",
    )
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.database_url == "postgresql://fish:secret@localhost/fish_memory"
    assert settings.llm_provider == "openai"
    assert settings.llm_api_key.get_secret_value() == "test-key"
    assert settings.llm_model == "qwen-plus"
    assert settings.llm_base_url == "https://example.invalid/v1"
    assert settings.answer_enable_search is False
    assert settings.embedding_provider == "local"
    assert settings.embedding_api_key is None
    assert settings.embedding_base_url is None
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.embedding_dimensions == 384
    assert settings.embedding_batch_size == 32
    assert settings.embedding_local_model_dir is None
    assert settings.embedding_local_num_threads == 2
    assert settings.embedding_local_max_tokens == 512
    assert settings.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_rejects_blank_llm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_LLM_API_KEY", "")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "FISH_MEMORY_LLM_API_KEY" in str(exc_info.value)


def test_api_auth_defaults_to_generated_file_backed_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.delenv("FISH_MEMORY_API_KEY", raising=False)
    monkeypatch.delenv("FISH_MEMORY_API_KEY_FILE", raising=False)

    settings = settings_without_env_file()

    assert settings.api_key is None
    assert settings.api_key_file == ".cache/fish-memory/api-key"


def test_reads_independent_remote_embedding_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    for name, value in EMBEDDING_ENV.items():
        monkeypatch.setenv(name, value)

    settings = settings_without_env_file()

    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_api_key is not None
    assert settings.embedding_api_key.get_secret_value() == "embedding-only-key"
    assert settings.embedding_base_url == "https://embedding.example.invalid/v1"
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.embedding_dimensions == 1024
    assert settings.embedding_batch_size == 16
    assert settings.llm_api_key.get_secret_value() == "test-key"
    assert settings.llm_base_url == "https://example.invalid/v1"


def test_accepts_non_1024_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_EMBEDDING_DIMENSIONS", "768")

    settings = settings_without_env_file()

    assert settings.embedding_dimensions == 768


def test_rejects_remote_embedding_without_independent_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_EMBEDDING_PROVIDER", "openai-compatible")
    monkeypatch.setenv(
        "FISH_MEMORY_EMBEDDING_BASE_URL",
        "https://embedding.example.invalid/v1",
    )

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "FISH_MEMORY_EMBEDDING_API_KEY" in str(exc_info.value)


def test_rejects_non_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MEMORY_DATABASE_URL", "postgresql://localhost/fish_memory")
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_LLM_PROVIDER", "anthropic")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "llm_provider" in str(exc_info.value)


def test_does_not_accept_legacy_hindsight_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in LEGACY_LLM_ENV.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "llm_provider" in str(exc_info.value)


def test_system_prompt_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MEMORY_DATABASE_URL", "postgresql://localhost/fish_memory")
    monkeypatch.setenv("FISH_MEMORY_SYSTEM_PROMPT", "你是一条安静的小鱼。")
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.system_prompt == "你是一条安静的小鱼。"


def test_answer_search_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_ANSWER_ENABLE_SEARCH", "false")

    settings = settings_without_env_file()

    assert settings.answer_enable_search is False


def test_background_settings_use_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)

    settings = settings_without_env_file()

    assert settings.background_poll_interval_seconds == 1.0
    assert settings.background_llm_concurrency == 1
    assert settings.background_embedding_concurrency == 4
    assert settings.fact_extraction_max_retries == 3
    assert settings.fact_llm_timeout_seconds == 120
    assert settings.fact_batch_size == 8
    assert settings.dynamic_snapshot_max_age_days == 7
    assert settings.snapshot_batch_size == 100
    assert settings.answer_history_message_limit == 50
    assert settings.history_token_limit == 20_000
    assert settings.history_token_retain == 10_000
    assert settings.assistant_api_url == "http://127.0.0.1:8000"
    assert settings.assistant_poll_interval_seconds == 1.0
    assert settings.asr_model == "sensevoice"
    assert settings.asr_num_threads == 2
    assert settings.companion_audio_block_size == 160
    assert settings.companion_aec_delay_ms == 50
    assert settings.companion_aec_delay_mode == "auto"
    assert settings.companion_vad_mode == 2
    assert settings.companion_vad_start_frames == 1
    assert settings.companion_vad_model_dir is None
    assert settings.companion_vad_threshold == 0.25
    assert settings.companion_vad_min_silence_ms == 350
    assert settings.companion_vad_min_speech_ms == 80
    assert settings.companion_vad_window_size == 512
    assert settings.companion_vad_max_speech_seconds == 20.0
    assert settings.companion_vad_num_threads == 1
    assert settings.companion_vad_end_silence_ms == 200
    assert settings.companion_utterance_commit_delay_ms == 0
    assert settings.companion_smart_turn_enabled is True
    assert settings.companion_smart_turn_model_path is None
    assert settings.companion_smart_turn_threshold == 0.5
    assert settings.companion_smart_turn_reopen_ms == 800
    assert settings.companion_smart_turn_timeout_ms == 500
    assert settings.companion_smart_turn_num_threads == 1
    assert settings.companion_early_interrupt_snapshot_ms == 300
    assert settings.companion_early_interrupt_min_chars == 2
    assert settings.companion_single_char_min_speech_ms == 200
    assert settings.audio_diagnostics_enabled is False
    assert settings.audio_diagnostics_dir == ".cache/fish-memory/audio-diagnostics"
    assert settings.audio_diagnostics_retention_days == 7
    assert settings.audio_diagnostics_pre_seconds == 5.0
    assert settings.audio_diagnostics_post_seconds == 5.0


@pytest.mark.parametrize("legacy_model", ["paraformer", "zipformer"])
def test_legacy_asr_model_is_normalized_to_sensevoice(
    monkeypatch: pytest.MonkeyPatch,
    legacy_model: str,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_ASR_MODEL", legacy_model)

    assert settings_without_env_file().asr_model == "sensevoice"


def test_reads_background_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    for name, value in BACKGROUND_ENV.items():
        monkeypatch.setenv(name, value)

    settings = settings_without_env_file()

    assert settings.background_poll_interval_seconds == 0.25
    assert settings.background_llm_concurrency == 4
    assert settings.background_embedding_concurrency == 6
    assert settings.fact_extraction_max_retries == 5
    assert settings.fact_llm_timeout_seconds == 120
    assert settings.fact_batch_size == 12
    assert settings.dynamic_snapshot_max_age_days == 9
    assert settings.snapshot_batch_size == 50
    assert settings.answer_history_message_limit == 50
    assert settings.history_token_limit == 160_000
    assert settings.history_token_retain == 110_000
    assert settings.assistant_api_url == "http://127.0.0.1:8100"
    assert settings.assistant_poll_interval_seconds == 2.5
    assert settings.asr_model == "sensevoice"
    assert settings.asr_num_threads == 3
    assert settings.companion_audio_block_size == 640
    assert settings.companion_audio_backend == "livekit"
    assert settings.livekit_audio_helper == "/tmp/fish-livekit-audio-helper"
    assert settings.companion_aec_delay_ms == 70
    assert settings.companion_aec_delay_mode == "auto"
    assert settings.companion_vad_mode == 3
    assert settings.companion_vad_start_frames == 3
    assert settings.companion_vad_model_dir == "/tmp/fish-vad"
    assert settings.companion_vad_threshold == 0.4
    assert settings.companion_vad_min_silence_ms == 420
    assert settings.companion_vad_min_speech_ms == 90
    assert settings.companion_vad_window_size == 512
    assert settings.companion_vad_max_speech_seconds == 30.0
    assert settings.companion_vad_num_threads == 2
    assert settings.companion_vad_end_silence_ms == 250
    assert settings.companion_utterance_commit_delay_ms == 180
    assert settings.companion_smart_turn_enabled is False
    assert settings.companion_smart_turn_model_path == "/tmp/smart-turn.onnx"
    assert settings.companion_smart_turn_threshold == 0.65
    assert settings.companion_smart_turn_reopen_ms == 900
    assert settings.companion_smart_turn_timeout_ms == 650
    assert settings.companion_smart_turn_num_threads == 2
    assert settings.companion_early_interrupt_snapshot_ms == 240
    assert settings.companion_early_interrupt_min_chars == 3
    assert settings.companion_single_char_min_speech_ms == 260
    assert settings.audio_diagnostics_enabled is True
    assert settings.audio_diagnostics_dir == "/tmp/fish-audio"
    assert settings.audio_diagnostics_retention_days == 14
    assert settings.audio_diagnostics_pre_seconds == 4.5
    assert settings.audio_diagnostics_post_seconds == 6.5


def test_rejects_invalid_aec_delay_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AEC_DELAY_MODE", "manual")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "companion_aec_delay_mode" in str(exc_info.value)


def test_rejects_invalid_companion_audio_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AUDIO_BACKEND", "automatic")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "companion_audio_backend" in str(exc_info.value)


@pytest.mark.parametrize(
    ("name", "value", "field"),
    [
        (
            "FISH_MEMORY_BACKGROUND_POLL_INTERVAL_SECONDS",
            "0",
            "background_poll_interval_seconds",
        ),
        (
            "FISH_MEMORY_BACKGROUND_LLM_CONCURRENCY",
            "0",
            "background_llm_concurrency",
        ),
        (
            "FISH_MEMORY_BACKGROUND_EMBEDDING_CONCURRENCY",
            "0",
            "background_embedding_concurrency",
        ),
        (
            "FISH_MEMORY_FACT_EXTRACTION_MAX_RETRIES",
            "0",
            "fact_extraction_max_retries",
        ),
        (
            "FISH_MEMORY_FACT_BATCH_SIZE",
            "0",
            "fact_batch_size",
        ),
        (
            "FISH_MEMORY_DYNAMIC_SNAPSHOT_MAX_AGE_DAYS",
            "0",
            "dynamic_snapshot_max_age_days",
        ),
        (
            "FISH_MEMORY_SNAPSHOT_BATCH_SIZE",
            "0",
            "snapshot_batch_size",
        ),
    ],
)
def test_rejects_invalid_background_settings(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    field: str,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert field in str(exc_info.value)


def test_rejects_history_retain_limit_not_below_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_llm_environment(monkeypatch)
    monkeypatch.setenv("FISH_MEMORY_HISTORY_TOKEN_LIMIT", "100000")
    monkeypatch.setenv("FISH_MEMORY_HISTORY_TOKEN_RETAIN", "100000")

    with pytest.raises(ValidationError) as exc_info:
        settings_without_env_file()

    assert "history_token_retain" in str(exc_info.value)
