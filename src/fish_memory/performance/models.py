from dataclasses import dataclass
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class PerformanceStep:
    name: str
    index: int
    started_offset_us: int
    duration_us: int
    input_payload: JsonValue = None
    output_payload: JsonValue = None
    error: str | None = None
