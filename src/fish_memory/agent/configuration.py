"""Resolve user-facing Agent URLs into internal durable targets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import SecretStr

from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.hermes import HermesConnectorFactory, is_loopback_host
from fish_memory.agent.models import AgentResolvedTarget
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.config import Settings
from fish_memory.db.agent_targets import TargetStore


def agent_on_this_computer(url: str | None) -> bool:
    """Whether the Agent at this URL runs on this computer, where its tools act."""
    try:
        host = urlsplit(url).hostname if url else None
    except ValueError:
        return False
    return host is not None and is_loopback_host(host)


@dataclass(frozen=True)
class GlobalAgentConfiguration:
    configured: bool
    credential_ref: str | None


async def configure_global_agent(
    settings: Settings,
    runtime: AgentRuntimeState,
    store: TargetStore,
    factory: HermesConnectorFactory,
    credentials: AgentCredentialResolver,
    *,
    enabled: bool,
    url: str,
    api_key: SecretStr | None,
) -> GlobalAgentConfiguration:
    """Update the local target and credential; never probe or restart a service."""
    reference = global_agent_credential_ref(factory, url) if api_key else None
    target = await ensure_global_agent_target(store, factory, url) if api_key else None
    if reference is not None and api_key is not None:
        credentials.store_global(reference, api_key.get_secret_value())
    else:
        credentials.clear_global_credentials()
    settings.agent_url, settings.agent_api_key = url, api_key
    settings.agent_tasks_enabled = enabled
    runtime.configure(
        enabled=enabled, default_target_id=target.target.id if target else None
    )
    return GlobalAgentConfiguration(target is not None, reference)


def global_agent_credential_ref(
    factory: HermesConnectorFactory,
    agent_url: str,
) -> str:
    digest = _agent_url_digest(factory, agent_url)
    return f"global:{digest}"


def fish_agent_credential_ref(
    factory: HermesConnectorFactory,
    fish_id: UUID,
    agent_url: str,
) -> str:
    digest = _agent_url_digest(factory, agent_url)
    return f"fish:{fish_id.hex}:{digest}"


def fish_agent_target_key(
    factory: HermesConnectorFactory,
    fish_id: UUID,
    agent_url: str,
    credential_ref: str,
) -> str:
    url_digest = _agent_url_digest(factory, agent_url)
    credential_digest = hashlib.sha256(credential_ref.encode("utf-8")).hexdigest()[:16]
    return f"fish:{fish_id.hex}:{url_digest}:{credential_digest}"


def _agent_url_digest(
    factory: HermesConnectorFactory,
    agent_url: str,
) -> str:
    target = factory.target_from_url(
        agent_url,
        credential_ref="global:probe",
        external_key="fish-memory:probe",
        display_name="Hermes",
    )
    canonical_url = (
        f"{target.endpoint.base_url.rstrip('/')}{target.target.route_prefix}"
    )
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:32]


async def ensure_global_agent_target(
    store: TargetStore,
    factory: HermesConnectorFactory,
    agent_url: str,
) -> AgentResolvedTarget:
    credential_ref = global_agent_credential_ref(factory, agent_url)
    target = factory.target_from_url(
        agent_url,
        credential_ref=credential_ref,
        external_key="fish-memory:global",
        display_name="Hermes",
    )
    return await store.upsert_target(
        str(target.endpoint.connector_type),
        target.endpoint.base_url,
        target.target.external_key,
        target.target.route_prefix,
        target.target.credential_ref,
        target.target.connector_config,
        target.target.display_name,
    )
