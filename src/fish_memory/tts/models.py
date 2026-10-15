from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TtsVoiceOption(BaseModel):
    id: str
    label: str


class TtsModelOption(BaseModel):
    id: str
    label: str
    voices: list[TtsVoiceOption]


class TtsStartRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    voice: str = Field(min_length=1, max_length=120)
    audio_format: Literal["pcm"] = "pcm"
    sample_rate: Literal[8_000, 16_000, 22_050, 24_000, 44_100, 48_000] = 22_050
    facial: bool = False
    speech_tags: bool = False


class TtsAppendRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    display_text: str | None = Field(default=None, max_length=20_000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class TtsRequest(TtsStartRequest):
    """Backward-compatible one-shot request used by provider helpers and callers."""

    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value
