"""The plugin's output contract, independent of its event input schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fish_memory.pet.types import PET_EMOTIONS


def expression_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "m": {"type": "string", "enum": sorted(PET_EMOTIONS)},
            "a": {"type": "string"},
        },
        "required": ["m", "a"],
        "additionalProperties": False,
    }


class InvalidExpression(ValueError):
    """A provider returned content outside the expression contract."""


class BriefEventSpeech(BaseModel):
    """Completed speech for durable, non-streaming host events."""

    model_config = ConfigDict(extra="forbid")
    m: Literal["neutral", "happy", "angry", "shy", "sad"]
    a: str = Field(min_length=1, max_length=800)
