from __future__ import annotations

from fish_memory.home.models import HomeEntity


def resolve_brightness_value(
    value: object,
    entity: HomeEntity,
    *,
    text: str = "",
) -> object:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        return value
    direction = _brightness_direction(value, text)
    if direction == 0:
        return value
    brightness = entity.attributes.get("brightness")
    if isinstance(brightness, bool) or not isinstance(brightness, (int, float)):
        raise ValueError("relative brightness requires current brightness")
    current = round(float(brightness) / 255 * 100)
    return min(100, max(0, current + direction * 10))


def _brightness_direction(value: object, text: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "increase":
            return 1
        if normalized == "decrease":
            return -1
    if any(marker in text for marker in ("调亮", "调高")):
        return 1
    if any(marker in text for marker in ("调暗", "调低")):
        return -1
    return 0
