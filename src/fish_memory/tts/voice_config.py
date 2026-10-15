from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from fish_memory.tts.models import TtsModelOption


class VoiceKind(StrEnum):
    PRESET = "preset"
    CUSTOM = "custom"
    LOCAL = "local"


class FishVoiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=120)
    voice: str = Field(min_length=1, max_length=160)
    kind: VoiceKind


@dataclass(frozen=True, slots=True)
class ResolvedVoiceConfig:
    config: FishVoiceConfig
    used_fallback: bool


def is_supported_voice(
    config: FishVoiceConfig,
    catalog: Iterable[TtsModelOption],
) -> bool:
    model = next((item for item in catalog if item.id == config.model), None)
    if model is None:
        return False
    if config.kind in {VoiceKind.PRESET, VoiceKind.LOCAL}:
        return any(option.id == config.voice for option in model.voices)
    # DashScope binds cloned voices to a target model and includes that model
    # in the returned ID. Enforcing the prefix prevents cross-model use.
    return config.voice.startswith(f"{config.model}-")


# 小新: the cloud voice a character speaks with until it picks its own.
DEFAULT_CLOUD_VOICE = "longanxiaoxin"


def resolve_voice_config(
    saved: FishVoiceConfig | None,
    *,
    provider: str,
) -> ResolvedVoiceConfig:
    from fish_memory.tts.catalog import get_local_tts_catalog, get_tts_catalog

    catalog = get_local_tts_catalog() if provider == "local" else get_tts_catalog()
    expected_kind = VoiceKind.LOCAL if provider == "local" else None
    if (
        saved is not None
        and is_supported_voice(saved, catalog)
        and (expected_kind is None or saved.kind is expected_kind)
        and (provider != "dashscope" or saved.kind is not VoiceKind.LOCAL)
    ):
        return ResolvedVoiceConfig(config=saved, used_fallback=False)
    model = catalog[0]
    voice = model.voices[0]
    if provider == "dashscope":
        preferred = next(
            (item for item in catalog if item.id == "qwen-audio-3.0-tts-flash"),
            model,
        )
        model = preferred
        voice = next(
            (item for item in model.voices if item.id == DEFAULT_CLOUD_VOICE),
            model.voices[0],
        )
    return ResolvedVoiceConfig(
        config=FishVoiceConfig(
            model=model.id,
            voice=voice.id,
            kind=(VoiceKind.LOCAL if provider == "local" else VoiceKind.PRESET),
        ),
        used_fallback=saved is not None,
    )
