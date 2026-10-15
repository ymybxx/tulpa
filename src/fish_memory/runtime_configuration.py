"""Authenticated desktop updates that do not rebuild memory or audio services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from fish_memory.config import Settings
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.service import ExperimentalService
from fish_memory.llm.configuration import DesktopModelRoutes
from fish_memory.llm.runtime import ModelRuntime
from fish_memory.tts.runtime import CustomVoiceRuntime

runtime_configuration_router = APIRouter(
    prefix="/internal/runtime", tags=["internal"], include_in_schema=False
)


class SpeechConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    provider: Literal["auto", "dashscope", "local"]
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def require_valid_key(self):
        if self.api_key is not None and not self.api_key.get_secret_value().strip():
            raise ValueError("speech API key cannot be blank")
        if self.provider == "dashscope" and self.api_key is None:
            raise ValueError("DashScope speech API key is required")
        return self


class AgentTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concurrency: int = Field(ge=1)
    poll_seconds: float = Field(gt=0)
    http_timeout_seconds: float = Field(gt=0)
    sse_read_timeout_seconds: float = Field(gt=0)
    reply_context_ttl_seconds: float = Field(gt=0)


class AgentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    enabled: bool = Field(strict=True)
    url: str = Field(min_length=1, max_length=512)
    api_key: SecretStr | None = None


class RuntimeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    models: DesktopModelRoutes
    speech: SpeechConfiguration
    voice_draft_enabled: bool
    speech_facial_wait_for_ready: bool | None = Field(default=None, strict=True)
    experimental_companion: ExperimentalConfiguration | None = None
    codex_notifications_enabled: bool | None = None
    # 联网查资料: on, and the reasoning model can search.
    web_search_enabled: bool | None = None
    microphone_enabled: bool | None = None
    playback_enabled: bool | None = None
    llm_log_retention_days: int | None = Field(default=None, strict=True, ge=1, le=365)
    agent_tuning: AgentTuning | None = None
    agent: AgentConfiguration | None = None
    forget_agent_credentials: bool = False


class RuntimeConfigurationService:
    def __init__(
        self,
        settings: Settings,
        models: ModelRuntime,
        custom_voices: CustomVoiceRuntime,
        apply_agent_tuning: Callable[[AgentTuning], None] | None = None,
        forget_agent_credentials: Callable[[], None] | None = None,
        apply_agent_configuration: Callable[[AgentConfiguration], Awaitable[None]]
        | None = None,
        configure_codex_notifications: Callable[[bool], None] | None = None,
        experiment: ExperimentalService | None = None,
        speech_configured: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.models = models
        self.custom_voices = custom_voices
        self.apply_agent_tuning = apply_agent_tuning
        self.forget_agent_credentials = forget_agent_credentials
        self.apply_agent_configuration = apply_agent_configuration
        self.configure_codex_notifications = configure_codex_notifications
        self.experiment = experiment
        self.speech_configured = speech_configured
        self._lock = asyncio.Lock()
        self.revision = 0

    async def apply(self, configuration: RuntimeConfiguration) -> dict[str, int]:
        async with self._lock:
            if configuration.agent is not None:
                if self.apply_agent_configuration is None:
                    raise HTTPException(409, "Task configuration is unavailable")
                await self.apply_agent_configuration(configuration.agent)
            self.models.configure(configuration.models)
            self.settings.desktop_model_routes = configuration.models
            if self.settings.tts_api_key != configuration.speech.api_key:
                self.custom_voices.invalidate()
            self.settings.tts_provider = configuration.speech.provider
            self.settings.tts_api_key = configuration.speech.api_key
            if self.speech_configured is not None:
                # The voice provider is known only now; prepare a local voice early.
                self.speech_configured()
            self.settings.companion_voice_draft_enabled = (
                configuration.voice_draft_enabled
            )
            if configuration.speech_facial_wait_for_ready is not None:
                self.settings.speech_facial_wait_for_ready = (
                    configuration.speech_facial_wait_for_ready
                )
            if (
                configuration.experimental_companion is not None
                and self.experiment is not None
            ):
                self.settings.experimental_companion = (
                    configuration.experimental_companion
                )
                self.experiment.configure(configuration.experimental_companion)
            if configuration.web_search_enabled is not None:
                self.settings.answer_enable_search = configuration.web_search_enabled
                if self.experiment is not None:
                    self.experiment.search_enabled = configuration.web_search_enabled
            if configuration.llm_log_retention_days is not None:
                self.settings.llm_log_retention_days = (
                    configuration.llm_log_retention_days
                )
            if configuration.codex_notifications_enabled is not None:
                self.settings.codex_notifications_enabled = (
                    configuration.codex_notifications_enabled
                )
                if self.configure_codex_notifications is not None:
                    self.configure_codex_notifications(
                        configuration.codex_notifications_enabled
                    )
            if (
                configuration.forget_agent_credentials
                and not self.settings.agent_tasks_enabled
                and self.settings.agent_api_key is not None
                and self.forget_agent_credentials is not None
            ):
                self.forget_agent_credentials()
                self.settings.agent_api_key = None
            if (
                configuration.agent_tuning is not None
                and self.apply_agent_tuning is not None
            ):
                self.apply_agent_tuning(configuration.agent_tuning)
            self.revision += 1
            return {"revision": self.revision}


@runtime_configuration_router.put("/configuration")
async def configure_runtime(
    body: RuntimeConfiguration, request: Request
) -> dict[str, int]:
    settings = request.app.state.settings
    if not settings.desktop_mode:
        raise HTTPException(404, "Desktop configuration is unavailable")
    service: RuntimeConfigurationService = request.app.state.runtime_configuration
    result = await service.apply(body)
    if body.microphone_enabled is not None:
        _configure_microphone(request, body.microphone_enabled)
    if body.playback_enabled is not None:
        _configure_playback(request, body.playback_enabled)
    return result


class MicrophoneConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)


@runtime_configuration_router.get("/codex-notifications")
async def codex_notification_status(request: Request) -> dict[str, object]:
    if not request.app.state.settings.desktop_mode:
        raise HTTPException(404, "Desktop configuration is unavailable")
    observer = getattr(request.app.state, "codex_completion_observer", None)
    return (
        observer.status() if observer else {"enabled": False, "error": "观察器不可用"}
    )


def _configure_microphone(request: Request, enabled: bool) -> dict[str, bool]:
    request.app.state.settings.companion_microphone_enabled = enabled
    request.app.state.pet_event_hub.publish_control(
        "pet.microphone", None, {"enabled": enabled}
    )
    return {"enabled": enabled}


@runtime_configuration_router.put("/microphone")
async def configure_microphone(
    body: MicrophoneConfiguration, request: Request
) -> dict[str, bool]:
    if not request.app.state.settings.desktop_mode:
        raise HTTPException(404, "Desktop configuration is unavailable")
    return _configure_microphone(request, body.enabled)


def _configure_playback(request: Request, enabled: bool) -> dict[str, bool]:
    request.app.state.settings.companion_playback_enabled = enabled
    request.app.state.pet_event_hub.publish_control(
        "pet.speaker", None, {"enabled": enabled}
    )
    return {"enabled": enabled}


@runtime_configuration_router.put("/playback")
async def configure_playback(
    body: MicrophoneConfiguration, request: Request
) -> dict[str, bool]:
    if not request.app.state.settings.desktop_mode:
        raise HTTPException(404, "Desktop configuration is unavailable")
    return _configure_playback(request, body.enabled)
