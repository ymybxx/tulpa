import re
from datetime import UTC, datetime
from typing import Literal, Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.set(current)

    def now(self) -> datetime:
        return self._current

    def set(self, current: datetime) -> None:
        if current.tzinfo is None:
            raise ValueError("current time must be timezone-aware")
        self._current = current.astimezone(UTC)


def format_utc_timestamp(value: datetime) -> str:
    """Render a compact UTC timestamp for persisted or machine-facing data."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def system_local_time(value: datetime) -> datetime:
    """Resolve the system timezone at this instant, including historical DST."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone()


def format_prompt_timestamp(
    value: datetime,
    *,
    timespec: Literal["hours", "minutes", "seconds"] = "seconds",
    sep: str = "T",
) -> str:
    """Render model-facing time in the user's current system timezone."""
    return system_local_time(value).isoformat(sep=sep, timespec=timespec)


def format_prompt_time_range(start: datetime, end: datetime) -> str:
    return f"{format_prompt_timestamp(start)} — {format_prompt_timestamp(end)}"


def format_prompt_timestamp_text(value: str) -> str:
    """Project a stored timestamp field; legacy naive fields were UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return format_prompt_timestamp(parsed)


_UTC_PASSAGE_RANGE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) — "
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC"
)


def localize_passage_timestamps(text: str) -> str:
    """Project a known provenance header, leaving every quoted word intact."""
    header, separator, words = text.partition("\n")
    match = _UTC_PASSAGE_RANGE.fullmatch(header)
    if match is None:
        return text
    return (
        " — ".join(format_prompt_timestamp_text(v) for v in match.groups())
        + separator
        + words
    )
