"""Stable semantic vocabulary for renderer-independent desktop-pet events."""

from enum import StrEnum


class PetEmotion(StrEnum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    HAPPY = "happy"
    SHY = "shy"
    SAD = "sad"


PET_EMOTIONS = frozenset(item.value for item in PetEmotion)
