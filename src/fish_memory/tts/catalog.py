from fish_memory.tts.models import TtsModelOption, TtsVoiceOption

_QWEN_VOICES = [
    TtsVoiceOption(id="longanfengyue", label="风悦"),
    TtsVoiceOption(id="longanyuanfei", label="远飞"),
    TtsVoiceOption(id="longanlingxi", label="灵犀"),
    TtsVoiceOption(id="longanxiaoxin", label="小新"),
    TtsVoiceOption(id="longanhuan_v3.6", label="欢欢"),
    TtsVoiceOption(id="longjielidou_v3.6", label="杰力豆"),
    TtsVoiceOption(id="longpaopao_v3.6", label="泡泡"),
    TtsVoiceOption(id="longhuohuo_v3.6", label="火火"),
    TtsVoiceOption(id="longchuanshu_v3.6", label="传书"),
    TtsVoiceOption(id="loongmary", label="Mary"),
    TtsVoiceOption(id="loongeva_v3.6", label="Eva"),
    TtsVoiceOption(id="loongjohn", label="John"),
]
_LOCAL_VOICE = TtsVoiceOption(id="default", label="默认中英双语音色")


_TTS_CATALOG = (
    TtsModelOption(
        id="qwen-audio-3.0-tts-flash",
        label="Qwen Audio TTS 3.0 Flash",
        voices=_QWEN_VOICES,
    ),
)

_LOCAL_TTS_CATALOG = (
    TtsModelOption(
        id="matcha-icefall-zh-en",
        label="Matcha Icefall 中英双语",
        voices=[_LOCAL_VOICE],
    ),
)


def get_tts_catalog() -> tuple[TtsModelOption, ...]:
    return _TTS_CATALOG


def get_local_tts_catalog() -> tuple[TtsModelOption, ...]:
    return _LOCAL_TTS_CATALOG


def find_tts_model(model_id: str) -> TtsModelOption | None:
    return next((model for model in _TTS_CATALOG if model.id == model_id), None)
