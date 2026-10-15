from __future__ import annotations

from typing import Any

import pytest

import fish_memory.db.pool as pool_module


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("desktop_mode", "expected_options"),
    [
        (False, {"dsn": "postgresql://test/fish"}),
        (
            True,
            {"dsn": "postgresql://test/fish", "min_size": 1, "max_size": 4},
        ),
    ],
)
async def test_pool_limits_are_desktop_only(
    monkeypatch: pytest.MonkeyPatch,
    desktop_mode: bool,
    expected_options: dict[str, object],
) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    async def fake_create_pool(**options: object) -> Any:
        calls.append(options)
        return sentinel

    monkeypatch.setattr(pool_module.asyncpg, "create_pool", fake_create_pool)

    result = await pool_module.create_pool(
        "postgresql://test/fish",
        desktop_mode=desktop_mode,
    )

    assert result is sentinel
    assert calls == [expected_options]
