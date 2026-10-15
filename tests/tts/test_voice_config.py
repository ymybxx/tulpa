from fish_memory.tts.catalog import find_tts_model, get_tts_catalog
from fish_memory.tts.voice_config import (
    FishVoiceConfig,
    VoiceKind,
    is_supported_voice,
    resolve_voice_config,
)


def test_qwen_flash_catalog_exposes_multiple_preset_voices() -> None:
    model = find_tts_model("qwen-audio-3.0-tts-flash")

    assert model is not None
    assert {voice.id for voice in model.voices} >= {
        "longanfengyue",
        "longanhuan_v3.6",
        "loongmary",
        "loongjohn",
    }


def test_remote_catalog_only_exposes_qwen_audio_flash() -> None:
    assert [model.id for model in get_tts_catalog()] == ["qwen-audio-3.0-tts-flash"]


def test_custom_voice_must_be_bound_to_selected_model() -> None:
    config = FishVoiceConfig(
        model="qwen-audio-3.0-tts-flash",
        voice="qwen-audio-3.0-tts-flash-myvoice-123",
        kind=VoiceKind.CUSTOM,
    )

    assert is_supported_voice(config, get_tts_catalog()) is True
    assert (
        is_supported_voice(
            config.model_copy(update={"voice": "cosyvoice-v3-flash-myvoice-123"}),
            get_tts_catalog(),
        )
        is False
    )


def test_resolve_voice_config_falls_back_when_provider_changes() -> None:
    saved = FishVoiceConfig(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        kind=VoiceKind.PRESET,
    )

    resolved = resolve_voice_config(saved, provider="local")

    assert resolved.config == FishVoiceConfig(
        model="matcha-icefall-zh-en",
        voice="default",
        kind=VoiceKind.LOCAL,
    )
    assert resolved.used_fallback is True


def test_cloud_default_voice_is_xiaoxin_until_a_character_chooses() -> None:
    resolved = resolve_voice_config(None, provider="dashscope")

    assert resolved.config.voice == "longanxiaoxin"
    assert resolved.config.kind is VoiceKind.PRESET
