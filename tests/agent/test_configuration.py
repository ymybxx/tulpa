from __future__ import annotations

from typing import Any

import pytest

from fish_memory.agent.configuration import (
    ensure_global_agent_target,
    global_agent_credential_ref,
)
from fish_memory.agent.hermes import HermesConnectorFactory


class RecordingTargetStore:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def upsert_target(self, *args: Any):  # type: ignore[no-untyped-def]
        self.calls.append(args)
        return args


@pytest.mark.asyncio
async def test_global_url_is_ensured_as_an_internal_default_target() -> None:
    store = RecordingTargetStore()
    factory = HermesConnectorFactory()
    credential_ref = global_agent_credential_ref(
        factory,
        "http://127.0.0.1:8642/p/coder/",
    )

    result = await ensure_global_agent_target(
        store,  # type: ignore[arg-type]
        factory,
        "http://127.0.0.1:8642/p/coder/",
    )

    assert result == store.calls[0]
    assert store.calls == [
        (
            "hermes",
            "http://127.0.0.1:8642",
            "fish-memory:global",
            "/p/coder",
            credential_ref,
            {},
            "Hermes",
        )
    ]


def test_global_credential_ref_is_stable_per_normalized_url() -> None:
    factory = HermesConnectorFactory()

    root = global_agent_credential_ref(factory, "http://127.0.0.1:8642/")
    same_root = global_agent_credential_ref(factory, "http://127.0.0.1:8642")
    coder = global_agent_credential_ref(
        factory,
        "http://127.0.0.1:8642/p/coder",
    )

    assert root == same_root
    assert root.startswith("global:")
    assert coder.startswith("global:")
    assert coder != root
