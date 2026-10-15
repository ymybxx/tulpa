from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.agent.configuration import global_agent_credential_ref
from fish_memory.agent.connector import (
    AgentCapabilities,
    AgentConnectorError,
    AgentConnectorFactory,
    AgentConnectorRegistry,
)
from fish_memory.agent.hermes import HermesConnectorFactory
from fish_memory.agent.models import (
    AgentEndpoint,
    AgentResolvedTarget,
    AgentTarget,
    ConnectorTypeSlug,
)
from fish_memory.agent.routes import agent_target_router

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
TARGET_ID = UUID("00000000-0000-0000-0000-000000000051")
NOW = datetime(2026, 8, 31, tzinfo=UTC)
GLOBAL_REF = global_agent_credential_ref(
    HermesConnectorFactory(),
    "http://127.0.0.1:8642",
)


def target(
    *,
    target_id: UUID = TARGET_ID,
    base_url: str = "http://127.0.0.1:8642",
    route_prefix: str = "",
    credential_ref: str = GLOBAL_REF,
) -> AgentResolvedTarget:
    endpoint_id = UUID(int=target_id.int + 100)
    return AgentResolvedTarget(
        target=AgentTarget(
            id=target_id,
            endpoint_id=endpoint_id,
            external_key="fish-memory:internal",
            route_prefix=route_prefix,
            credential_ref=credential_ref,
            connector_config={},
            display_name="Hermes",
            enabled=True,
            created_at=NOW,
            updated_at=NOW,
        ),
        endpoint=AgentEndpoint(
            id=endpoint_id,
            connector_type=ConnectorTypeSlug("hermes"),
            base_url=base_url,
            created_at=NOW,
            updated_at=NOW,
        ),
    )


class ConfigTargetStore:
    def __init__(self) -> None:
        self.override: AgentResolvedTarget | None = None
        self.targets: dict[UUID, AgentResolvedTarget] = {}
        self.upsert_calls: list[tuple[object, ...]] = []
        self.set_calls: list[tuple[UUID, UUID | None]] = []
        self.fish_exists_result = True
        self.set_result = True

    async def get_fish_target(self, fish_id: UUID) -> AgentResolvedTarget | None:
        assert fish_id == FISH_ID
        return self.override

    async def fish_exists(self, fish_id: UUID) -> bool:
        assert fish_id == FISH_ID
        return self.fish_exists_result

    async def upsert_target(self, *args: object) -> AgentResolvedTarget:
        self.upsert_calls.append(args)
        created = target(
            target_id=UUID(int=80 + len(self.upsert_calls)),
            base_url=cast(str, args[1]),
            route_prefix=cast(str, args[3]),
            credential_ref=cast(str, args[4]),
        )
        self.targets[created.target.id] = created
        return created

    async def set_fish_target(self, fish_id: UUID, target_id: UUID | None) -> bool:
        self.set_calls.append((fish_id, target_id))
        if not self.fish_exists_result or not self.set_result:
            return False
        if target_id is None:
            self.override = None
        else:
            self.override = self.targets[target_id]
        return True


class ConfigCredentialResolver:
    def __init__(self) -> None:
        self.values = {GLOBAL_REF: "global-key"}
        self.transient: dict[str, str] = {}
        self.stored: list[tuple[str, str]] = []
        self.removed: list[str] = []
        self.locks: dict[str, asyncio.Lock] = {}

    def resolve(self, credential_ref: str) -> str:
        if credential_ref in self.transient:
            return self.transient[credential_ref]
        return self.values[credential_ref]

    def resolve_optional(self, credential_ref: str) -> str | None:
        try:
            return self.resolve(credential_ref)
        except KeyError:
            return None

    def store(self, credential_ref: str, secret: str) -> None:
        self.values[credential_ref] = secret.strip()
        self.stored.append((credential_ref, secret.strip()))

    def remove(self, credential_ref: str) -> None:
        self.values.pop(credential_ref, None)
        self.removed.append(credential_ref)

    def stage(self, credential_ref: str, secret: str) -> None:
        self.transient[credential_ref] = secret

    def unstage(self, credential_ref: str) -> None:
        self.transient.pop(credential_ref, None)

    @asynccontextmanager
    async def update_lock(self, credential_ref: str):  # type: ignore[no-untyped-def]
        lock = self.locks.setdefault(credential_ref, asyncio.Lock())
        async with lock:
            yield


class ConfigConnector:
    def __init__(self, credentials: ConfigCredentialResolver) -> None:
        self.credentials = credentials
        self.probed: list[AgentResolvedTarget] = []
        self.probed_keys: list[str] = []
        self.error: AgentConnectorError | None = None
        self.fail_keys: set[str] = set()
        self.block_key: str | None = None
        self.block_started = asyncio.Event()
        self.block_release = asyncio.Event()

    async def probe(self, resolved: AgentResolvedTarget) -> AgentCapabilities:
        key = self.credentials.resolve(cast(str, resolved.target.credential_ref))
        self.probed.append(resolved)
        self.probed_keys.append(key)
        if key == self.block_key:
            self.block_started.set()
            await self.block_release.wait()
        if self.error is not None:
            raise self.error
        if key in self.fail_keys:
            raise AgentConnectorError("probe failed", code="endpoint_unavailable")
        return AgentCapabilities(True, True, True, True, True, True, True)


def configuration_app() -> tuple[
    FastAPI,
    ConfigTargetStore,
    ConfigCredentialResolver,
    ConfigConnector,
]:
    store = ConfigTargetStore()
    credentials = ConfigCredentialResolver()
    connector = ConfigConnector(credentials)
    factory = HermesConnectorFactory()
    registry = AgentConnectorRegistry()
    registry.register(cast(AgentConnectorFactory, factory))
    app = FastAPI()
    app.state.agent_target_store = store
    app.state.agent_connector_registry = registry
    app.state.agent_connectors = {"hermes": connector}
    app.state.agent_credential_resolver = credentials
    app.state.agent_default_target_id = TARGET_ID
    app.state.agent_global_url = "http://127.0.0.1:8642"
    app.state.agent_global_configured = True
    app.state.agent_global_credential_ref = GLOBAL_REF
    app.state.agent_configuration_writable = True
    app.state.settings = SimpleNamespace(agent_tasks_enabled=True)
    app.include_router(agent_target_router)
    return app, store, credentials, connector


@pytest.mark.asyncio
async def test_reads_effective_fish_config_without_internal_target_fields() -> None:
    app, _, _, _ = configuration_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/internal/fishes/{FISH_ID}/agent-config")

    assert response.status_code == 200
    assert response.json() == {
        "agentTasksEnabled": True,
        "configurationWritable": True,
        "globalConfigured": True,
        "globalUrl": "http://127.0.0.1:8642",
        "inheritsGlobal": True,
        "url": "http://127.0.0.1:8642",
        "hasApiKeyOverride": False,
    }
    assert "target" not in response.text.lower()
    assert "credential" not in response.text.lower()


@pytest.mark.asyncio
async def test_disabled_runtime_rejects_fish_config_writes_without_side_effects() -> (
    None
):
    app, store, credentials, connector = configuration_app()
    app.state.agent_configuration_writable = False
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        current = await client.get(f"/internal/fishes/{FISH_ID}/agent-config")
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={
                "inheritGlobal": False,
                "url": "http://127.0.0.1:8642/p/coder",
                "apiKey": "private-key",
            },
        )

    assert current.json()["configurationWritable"] is False
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "agent_tasks_disabled"
    assert credentials.stored == []
    assert connector.probed == []
    assert store.upsert_calls == []


@pytest.mark.asyncio
async def test_saves_complete_url_and_key_as_an_opaque_fish_override() -> None:
    app, store, credentials, connector = configuration_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={
                "inheritGlobal": False,
                "url": "http://127.0.0.1:8642/p/coder/",
                "apiKey": "private-coder-key",
            },
        )

    assert response.status_code == 200
    assert response.json()["url"] == "http://127.0.0.1:8642/p/coder"
    assert response.json()["hasApiKeyOverride"] is True
    assert "private-coder-key" not in response.text
    assert len(credentials.stored) == 1
    credential_ref, secret = credentials.stored[0]
    assert credential_ref.startswith("fish:")
    assert secret == "private-coder-key"
    assert connector.probed[0].endpoint.base_url == "http://127.0.0.1:8642"
    assert connector.probed[0].target.route_prefix == "/p/coder"
    assert store.upsert_calls[0][4] == credential_ref
    assert store.set_calls == [(FISH_ID, store.override.target.id)]  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_rotating_one_fish_url_reuses_its_opaque_internal_identity() -> None:
    app, store, credentials, _ = configuration_app()
    transport = httpx.ASGITransport(app=app)
    body = {
        "inheritGlobal": False,
        "url": "http://127.0.0.1:8642/p/coder",
        "apiKey": "first-key",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json=body,
        )
        body["apiKey"] = "rotated-key"
        second = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json=body,
        )

    assert first.status_code == second.status_code == 200
    assert credentials.stored[0][0] == credentials.stored[1][0]
    assert store.upsert_calls[0][2] == store.upsert_calls[1][2]
    assert credentials.values[credentials.stored[0][0]] == "rotated-key"


@pytest.mark.asyncio
async def test_failed_rotation_keeps_the_previous_live_fish_key() -> None:
    app, _, credentials, connector = configuration_app()
    transport = httpx.ASGITransport(app=app)
    url = "http://127.0.0.1:8642/p/coder"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url, "apiKey": "working-key"},
        )
        connector.fail_keys.add("candidate-key")
        failed = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url, "apiKey": "candidate-key"},
        )

    assert first.status_code == 200
    assert failed.status_code == 502
    stable_ref = credentials.stored[0][0]
    assert credentials.values[stable_ref] == "working-key"
    assert connector.probed_keys == ["working-key", "candidate-key"]


@pytest.mark.asyncio
async def test_concurrent_failed_and_successful_rotations_keep_the_successful_key() -> (
    None
):
    app, _, credentials, connector = configuration_app()
    transport = httpx.ASGITransport(app=app)
    url = "http://127.0.0.1:8642/p/coder"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        initial = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url, "apiKey": "working-key"},
        )
        assert initial.status_code == 200
        connector.block_key = "bad-key"
        connector.fail_keys.add("bad-key")
        bad_request = asyncio.create_task(
            client.put(
                f"/internal/fishes/{FISH_ID}/agent-config",
                json={"inheritGlobal": False, "url": url, "apiKey": "bad-key"},
            )
        )
        await connector.block_started.wait()
        good_request = asyncio.create_task(
            client.put(
                f"/internal/fishes/{FISH_ID}/agent-config",
                json={"inheritGlobal": False, "url": url, "apiKey": "good-key"},
            )
        )
        await asyncio.sleep(0)
        assert not good_request.done()
        connector.block_release.set()
        bad, good = await asyncio.gather(bad_request, good_request)

    assert bad.status_code == 502
    assert good.status_code == 200
    stable_ref = credentials.stored[0][0]
    assert credentials.values[stable_ref] == "good-key"


@pytest.mark.asyncio
async def test_same_origin_override_can_inherit_the_global_key() -> None:
    app, store, credentials, connector = configuration_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={
                "inheritGlobal": False,
                "url": "http://127.0.0.1:8642/p/coder",
            },
        )

    assert response.status_code == 200
    assert credentials.stored == []
    assert connector.probed[0].target.credential_ref == GLOBAL_REF
    assert store.upsert_calls[0][4] == GLOBAL_REF


@pytest.mark.asyncio
async def test_switching_from_global_to_private_key_creates_a_new_target_identity() -> (
    None
):
    app, store, _, _ = configuration_app()
    transport = httpx.ASGITransport(app=app)
    url = "http://127.0.0.1:8642/p/coder"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        inherited = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url},
        )
        private = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url, "apiKey": "private-key"},
        )

    assert inherited.status_code == private.status_code == 200
    assert store.upsert_calls[0][2] != store.upsert_calls[1][2]
    assert store.upsert_calls[0][4] == GLOBAL_REF
    assert cast(str, store.upsert_calls[1][4]).startswith("fish:")


@pytest.mark.asyncio
async def test_failed_private_switch_leaves_previous_global_target_untouched() -> None:
    app, store, credentials, _ = configuration_app()
    transport = httpx.ASGITransport(app=app)
    url = "http://127.0.0.1:8642/p/coder"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        inherited = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url},
        )
        previous_target = store.override
        store.set_result = False
        failed = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": url, "apiKey": "private-key"},
        )

    assert inherited.status_code == 200
    assert failed.status_code == 404
    assert store.override == previous_target
    assert previous_target is not None
    assert previous_target.target.credential_ref == GLOBAL_REF
    assert store.upsert_calls[0][2] != store.upsert_calls[1][2]
    private_ref = cast(str, store.upsert_calls[1][4])
    assert private_ref not in credentials.values


@pytest.mark.asyncio
async def test_different_origin_requires_a_new_key_before_any_side_effect() -> None:
    app, store, credentials, connector = configuration_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": False, "url": "https://agent.example.test"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "agent_api_key_required"
    assert credentials.stored == []
    assert connector.probed == []
    assert store.upsert_calls == []
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_probe_failure_rolls_back_the_staged_secret_and_target_write() -> None:
    app, store, credentials, connector = configuration_app()
    connector.error = AgentConnectorError(
        "Bearer private-coder-key",
        code="endpoint_unavailable",
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={
                "inheritGlobal": False,
                "url": "http://127.0.0.1:8642/p/coder",
                "apiKey": "private-coder-key",
            },
        )

    assert response.status_code == 502
    assert "private-coder-key" not in response.text
    assert "Target" not in response.text
    assert credentials.stored == []
    assert credentials.removed == []
    assert credentials.transient == {}
    assert store.upsert_calls == []
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_clearing_override_returns_to_the_global_config() -> None:
    app, store, _, _ = configuration_app()
    store.override = target(
        route_prefix="/p/coder",
        credential_ref="fish:old-override",
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/internal/fishes/{FISH_ID}/agent-config",
            json={"inheritGlobal": True},
        )

    assert response.status_code == 200
    assert response.json()["inheritsGlobal"] is True
    assert store.set_calls == [(FISH_ID, None)]
