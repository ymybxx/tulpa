from datetime import UTC, datetime

import pytest

from fish_memory.clock import MutableClock, SystemClock, format_utc_timestamp


def test_system_clock_returns_aware_utc_time() -> None:
    value = SystemClock().now()

    assert value.tzinfo is UTC


def test_mutable_clock_returns_and_updates_current_time() -> None:
    initial = datetime(2023, 5, 20, 2, 21, tzinfo=UTC)
    updated = datetime(2024, 6, 1, 9, 30, tzinfo=UTC)
    clock = MutableClock(initial)

    assert clock.now() == initial

    clock.set(updated)

    assert clock.now() == updated


def test_mutable_clock_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MutableClock(datetime(2023, 5, 20, 2, 21))


def test_format_utc_timestamp_uses_second_precision() -> None:
    value = datetime(2026, 9, 4, 7, 50, 40, 687000, tzinfo=UTC)

    assert format_utc_timestamp(value) == "2026-09-04T07:50:40+00:00"

    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc_timestamp(value.replace(tzinfo=None))
