"""Persistence for configured external Agent targets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentResolvedTarget,
    AgentTarget,
    ConnectorTypeSlug,
)
from fish_memory.db.json_values import canonical_json_object

_MAX_EXTERNAL_ID_CHARS = 512
_MAX_CONNECTOR_CONFIG_BYTES = 16_384

_RESOLVED_TARGET_COLUMNS = """
target.id AS target_id,
target.endpoint_id AS target_endpoint_id,
target.external_key AS target_external_key,
target.route_prefix AS target_route_prefix,
target.credential_ref AS target_credential_ref,
target.connector_config AS target_connector_config,
target.display_name AS target_display_name,
target.enabled AS target_enabled,
target.created_at AS target_created_at,
target.updated_at AS target_updated_at,
endpoint.id AS endpoint_id,
endpoint.connector_type AS endpoint_connector_type,
endpoint.base_url AS endpoint_base_url,
endpoint.created_at AS endpoint_created_at,
endpoint.updated_at AS endpoint_updated_at
"""


class TargetStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_target(
        self,
        connector_type: str,
        base_url: str,
        external_key: str,
        route_prefix: str,
        credential_ref: str | None,
        connector_config: Mapping[str, object] | None,
        display_name: str,
    ) -> AgentResolvedTarget:
        slug = ConnectorTypeSlug(connector_type)
        _bounded_string("base_url", base_url, _MAX_EXTERNAL_ID_CHARS)
        _bounded_string("external_key", external_key, _MAX_EXTERNAL_ID_CHARS)
        _bounded_string("route_prefix", route_prefix, _MAX_EXTERNAL_ID_CHARS)
        _bounded_string("display_name", display_name, 200)
        if credential_ref is not None:
            _bounded_string("credential_ref", credential_ref, _MAX_EXTERNAL_ID_CHARS)
        config_json = _connector_config_json(connector_config)
        endpoint_id = uuid4()
        target_id = uuid4()
        row = await self._pool.fetchrow(
            f"""
            WITH endpoint AS (
                INSERT INTO agent_endpoints (id, connector_type, base_url)
                VALUES ($1, $2, $3)
                ON CONFLICT (connector_type, base_url) DO UPDATE
                SET updated_at = date_trunc('second', now())
                RETURNING *
            ),
            target AS (
                INSERT INTO agent_targets (
                    id,
                    endpoint_id,
                    external_key,
                    route_prefix,
                    credential_ref,
                    connector_config,
                    display_name,
                    enabled
                )
                SELECT $4, endpoint.id, $5, $6, $7, $8::jsonb, $9, true
                FROM endpoint
                ON CONFLICT (endpoint_id, external_key, route_prefix) DO UPDATE
                SET credential_ref = EXCLUDED.credential_ref,
                    connector_config = EXCLUDED.connector_config,
                    display_name = EXCLUDED.display_name,
                    enabled = true,
                    updated_at = date_trunc('second', now())
                RETURNING *
            )
            SELECT {_RESOLVED_TARGET_COLUMNS}
            FROM target
            JOIN endpoint ON endpoint.id = target.endpoint_id
            """,
            endpoint_id,
            str(slug),
            base_url,
            target_id,
            external_key,
            route_prefix,
            credential_ref,
            config_json,
            display_name,
        )
        return _required_resolved_target(row)

    async def list_targets(self) -> Sequence[AgentResolvedTarget]:
        rows = await self._pool.fetch(
            f"""
            SELECT {_RESOLVED_TARGET_COLUMNS}
            FROM agent_targets AS target
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            ORDER BY target.display_name, target.id
            """
        )
        return tuple(_resolved_target_from_record(row) for row in rows)

    async def resolve_for_fish(
        self,
        fish_id: UUID,
        default_target_id: UUID | None,
    ) -> AgentResolvedTarget | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_RESOLVED_TARGET_COLUMNS}
            FROM fishes AS fish
            JOIN agent_targets AS target
              ON target.id = COALESCE(fish.agent_target_id, $2)
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE fish.id = $1
              AND fish.deleted_at IS NULL
              AND target.enabled
            """,
            fish_id,
            default_target_id,
        )
        return None if row is None else _resolved_target_from_record(row)

    async def get_fish_target(
        self,
        fish_id: UUID,
    ) -> AgentResolvedTarget | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_RESOLVED_TARGET_COLUMNS}
            FROM fishes AS fish
            JOIN agent_targets AS target
              ON target.id = fish.agent_target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE fish.id = $1
              AND fish.deleted_at IS NULL
            """,
            fish_id,
        )
        return None if row is None else _resolved_target_from_record(row)

    async def fish_exists(self, fish_id: UUID) -> bool:
        row = await self._pool.fetchrow(
            """
            SELECT fish.id
            FROM fishes AS fish
            WHERE fish.id = $1
              AND fish.deleted_at IS NULL
            """,
            fish_id,
        )
        return row is not None

    async def set_fish_target(
        self,
        fish_id: UUID,
        target_id: UUID | None,
    ) -> bool:
        row = await self._pool.fetchrow(
            """
            UPDATE fishes AS fish
            SET agent_target_id = $2
            WHERE fish.id = $1
              AND fish.deleted_at IS NULL
              AND (
                  $2::uuid IS NULL
                  OR EXISTS (
                      SELECT 1
                      FROM agent_targets AS target
                      WHERE target.id = $2
                        AND target.enabled
                  )
              )
            RETURNING fish.id
            """,
            fish_id,
            target_id,
        )
        return row is not None


def _connector_config_json(config: Mapping[str, object] | None) -> str:
    payload = canonical_json_object(
        config,
        label="connector_config",
        maximum_bytes=_MAX_CONNECTOR_CONFIG_BYTES,
        none_as_empty=True,
    )
    if payload is None:
        raise RuntimeError("connector_config normalization returned no JSON")
    return payload


def _bounded_string(name: str, value: str, maximum: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip() and name not in {"route_prefix"}:
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")


def _required_resolved_target(
    record: Mapping[str, object] | None,
) -> AgentResolvedTarget:
    if record is None:
        raise RuntimeError("database did not return the Agent target")
    return _resolved_target_from_record(record)


def _resolved_target_from_record(record: Mapping[str, object]) -> AgentResolvedTarget:
    raw_config = record["target_connector_config"]
    if raw_config is None:
        config: Mapping[str, object] = {}
    elif isinstance(raw_config, str):
        decoded = json.loads(raw_config)
        config = cast(Mapping[str, object], decoded)
    else:
        config = cast(Mapping[str, object], raw_config)
    return AgentResolvedTarget(
        target=AgentTarget(
            id=cast(UUID, record["target_id"]),
            endpoint_id=cast(UUID, record["target_endpoint_id"]),
            external_key=cast(str, record["target_external_key"]),
            route_prefix=cast(str, record["target_route_prefix"]),
            credential_ref=cast(str | None, record["target_credential_ref"]),
            connector_config=config,
            display_name=cast(str, record["target_display_name"]),
            enabled=cast(bool, record["target_enabled"]),
            created_at=cast(datetime, record["target_created_at"]),
            updated_at=cast(datetime, record["target_updated_at"]),
        ),
        endpoint=AgentEndpoint(
            id=cast(UUID, record["endpoint_id"]),
            connector_type=ConnectorTypeSlug(
                cast(str, record["endpoint_connector_type"])
            ),
            base_url=cast(str, record["endpoint_base_url"]),
            created_at=cast(datetime, record["endpoint_created_at"]),
            updated_at=cast(datetime, record["endpoint_updated_at"]),
        ),
    )
