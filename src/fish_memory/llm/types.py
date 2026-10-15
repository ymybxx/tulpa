from dataclasses import dataclass

from fish_memory.performance.models import JsonValue


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    content: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    search_diagnostics: dict[str, JsonValue] | None = None
