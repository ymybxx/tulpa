from __future__ import annotations

import importlib
import importlib.metadata
from typing import cast

import pytest

from fish_memory.agent.connector import (
    AGENT_CONNECTOR_CONTRACT_VERSION,
    AgentConnectorError,
    AgentConnectorFactory,
    AgentConnectorRegistry,
    ConnectorDescriptor,
)


class _FakeFactory:
    def __init__(self, slug: str = "fake", version: int = 1) -> None:
        self.descriptor = ConnectorDescriptor(
            type_slug=slug,
            contract_version=version,
            display_name="Fake Connector",
        )

    def validate_target(self, target: object) -> object:
        return target

    def create(self, runtime: object) -> object:
        return runtime


def test_registry_registers_and_requires_a_real_factory() -> None:
    registry = AgentConnectorRegistry()
    factory = cast(AgentConnectorFactory, _FakeFactory("fixture-agent"))

    registry.register(factory)

    assert registry.require("fixture-agent") is factory


def test_registry_rejects_duplicate_slug() -> None:
    registry = AgentConnectorRegistry()
    registry.register(cast(AgentConnectorFactory, _FakeFactory("duplicate")))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(cast(AgentConnectorFactory, _FakeFactory("duplicate")))


def test_registry_rejects_incompatible_contract_version() -> None:
    registry = AgentConnectorRegistry()

    with pytest.raises(ValueError, match="contract version"):
        registry.register(
            cast(
                AgentConnectorFactory,
                _FakeFactory("future", AGENT_CONNECTOR_CONTRACT_VERSION + 1),
            )
        )


def test_registry_rejects_invalid_connector_slug() -> None:
    registry = AgentConnectorRegistry()

    with pytest.raises(ValueError, match="connector type"):
        registry.register(cast(AgentConnectorFactory, _FakeFactory("Unsafe/Import")))


def test_unknown_slug_fails_closed_only_when_required() -> None:
    registry = AgentConnectorRegistry()
    registry.register(cast(AgentConnectorFactory, _FakeFactory("known")))

    with pytest.raises(AgentConnectorError) as exc_info:
        registry.require("configured-but-not-installed")

    assert exc_info.value.code == "unsupported_capability"
    assert exc_info.value.retryable is False
    assert exc_info.value.dispatch_uncertain is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_registry_never_scans_entry_points_or_imports_provider_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dynamic_import(*args: object, **kwargs: object) -> object:
        raise AssertionError("registry attempted dynamic discovery")

    monkeypatch.setattr(importlib, "import_module", fail_dynamic_import)
    monkeypatch.setattr(importlib.metadata, "entry_points", fail_dynamic_import)
    registry = AgentConnectorRegistry()
    factory = cast(AgentConnectorFactory, _FakeFactory("static-only"))

    registry.register(factory)

    assert registry.require("static-only") is factory


def test_normalized_error_fields_are_bounded_at_the_contract_boundary() -> None:
    error = AgentConnectorError(
        "x" * 5_000,
        code="provider_error",
        retry_after="y" * 200,
    )

    assert len(str(error)) == 4_000
    assert error.retry_after is not None
    assert len(error.retry_after) == 128
