"""Low-cost transcript admission rules for companion voice turns."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, TypeAlias

UtteranceRejectionReason: TypeAlias = Literal[
    "non_speech_event",
    "nonlexical_vocalization",
]

_NON_SPEECH_EVENT_LABELS = frozenset({"bgm", "breath", "cough", "sneeze"})
_NONLEXICAL_VOCALIZATION_CHARACTERS = frozenset({"嗯", "哼", "咳"})


def compact_utterance_text(value: str) -> str:
    """Normalize text and remove Unicode whitespace and punctuation."""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character.casefold()
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def classify_utterance(
    text: str,
    asr_event: str = "",
    *,
    allow_nonlexical: bool = False,
) -> UtteranceRejectionReason | None:
    """Return a narrow rejection reason, or ``None`` when this gate admits it.

    Explicit SenseVoice non-speech metadata remains authoritative. The optional
    exemption applies only to ambiguous vocalizations such as a standalone
    ``嗯`` in a context that explicitly expects a short acknowledgement.
    """
    labels = {label.casefold() for label in re.findall(r"[A-Za-z]+", asr_event)}
    if "speech" not in labels and labels & _NON_SPEECH_EVENT_LABELS:
        return "non_speech_event"

    compact = compact_utterance_text(text)
    if (
        compact
        and set(compact) <= _NONLEXICAL_VOCALIZATION_CHARACTERS
        and not allow_nonlexical
    ):
        return "nonlexical_vocalization"
    return None


def low_evidence_utterance_reason(
    text: str,
    event: str = "",
    *,
    allow_nonlexical: bool = False,
) -> UtteranceRejectionReason | None:
    """Compatibility entry point for voice-pipeline admission gates."""
    return classify_utterance(
        text,
        event,
        allow_nonlexical=allow_nonlexical,
    )
