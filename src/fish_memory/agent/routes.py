"""Fish-Session-scoped durable Agent Message delivery and consumption routes."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from fish_memory.agent.configuration import (
    fish_agent_credential_ref,
    fish_agent_target_key,
)
from fish_memory.agent.connector import (
    AgentConnector,
    AgentConnectorError,
    AgentConnectorRegistry,
)
from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.hermes import HermesConnectorFactory
from fish_memory.agent.models import (
    AgentMessageEnvelope,
    AgentResolvedTarget,
)
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import (
    AgentMessageSubjectConflictError,
    TaskStore,
)
from fish_memory.db.sessions import SessionNotFoundError, SessionStore

agent_router = APIRouter(prefix="/v1/fishes/{fish_id}/agent-messages", tags=["public"])
agent_target_router = APIRouter(tags=["internal"], include_in_schema=False)

_SSE_HEARTBEAT_SECONDS = 15.0
_POLL_INTERVAL_SECONDS = 0.4
_INCOMPLETE_SCAN_INTERVAL_SECONDS = 2.0
_DELIVERY_PAGE_SIZE = 100
_MAX_NEW_PAGES_PER_CYCLE = 16
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_NON_NEGATIVE_INTEGER_RE = re.compile(r"^[0-9]+$")


class AgentMessageConsumptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    consumed_offset: int = Field(
        validation_alias="consumedOffset",
        serialization_alias="consumedOffset",
        ge=0,
        strict=True,
    )
    interrupted: bool = Field(strict=True)


class FishAgentConfigurationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    inherit_global: bool = Field(validation_alias="inheritGlobal", strict=True)
    url: str | None = Field(default=None, max_length=512, strict=True)
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="apiKey",
        max_length=8_192,
    )


class FishAgentConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    agent_tasks_enabled: bool = Field(serialization_alias="agentTasksEnabled")
    configuration_writable: bool = Field(serialization_alias="configurationWritable")
    global_configured: bool = Field(serialization_alias="globalConfigured")
    global_url: str | None = Field(serialization_alias="globalUrl")
    inherits_global: bool = Field(serialization_alias="inheritsGlobal")
    url: str | None
    has_api_key_override: bool = Field(serialization_alias="hasApiKeyOverride")


@agent_target_router.get(
    "/internal/fishes/{fish_id}/agent-config",
    response_model=FishAgentConfigurationResponse,
    response_model_by_alias=True,
)
async def get_fish_agent_configuration(
    fish_id: UUID,
    request: Request,
) -> FishAgentConfigurationResponse | JSONResponse:
    store = cast(TargetStore, request.app.state.agent_target_store)
    override = await store.get_fish_target(fish_id)
    if override is None and not await store.fish_exists(fish_id):
        return _target_error(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "Fish 不存在",
        )
    return _fish_agent_configuration_response(request, override)


@agent_target_router.put(
    "/internal/fishes/{fish_id}/agent-config",
    response_model=FishAgentConfigurationResponse,
    response_model_by_alias=True,
)
async def put_fish_agent_configuration(
    fish_id: UUID,
    request: Request,
) -> FishAgentConfigurationResponse | JSONResponse:
    if not bool(getattr(request.app.state, "agent_configuration_writable", False)):
        return _target_error(
            status.HTTP_409_CONFLICT,
            "agent_tasks_disabled",
            "请先启用 Agent 任务",
        )
    body = await _parse_fish_agent_configuration(request)
    if isinstance(body, JSONResponse):
        return body
    store = cast(TargetStore, request.app.state.agent_target_store)
    if not await store.fish_exists(fish_id):
        return _target_error(
            status.HTTP_404_NOT_FOUND,
            "fish_not_found",
            "Fish 不存在",
        )
    if body.inherit_global:
        if body.url is not None or body.api_key is not None:
            return _target_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "invalid_agent_configuration",
                "跟随全局配置时不能同时提交覆盖字段",
            )
        if not await store.set_fish_target(fish_id, None):
            return _target_error(
                status.HTTP_404_NOT_FOUND,
                "fish_not_found",
                "Fish 不存在",
            )
        return _fish_agent_configuration_response(request, None)

    if body.url is None or not body.url.strip():
        return _target_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "agent_url_required",
            "自定义 Agent 配置必须填写 URL",
        )
    registry = cast(
        AgentConnectorRegistry,
        request.app.state.agent_connector_registry,
    )
    connectors = cast(
        Mapping[str, AgentConnector],
        request.app.state.agent_connectors,
    )
    try:
        factory = cast(HermesConnectorFactory, registry.require("hermes"))
        candidate_for_origin = factory.target_from_url(
            body.url,
            credential_ref="global:probe",
            external_key="fish-memory:probe",
            display_name="Hermes",
        )
    except AgentConnectorError as error:
        return _target_connector_error(error, probing=False)
    connector = connectors.get("hermes")
    if connector is None:
        return _target_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "agent_connector_unavailable",
            "Agent Connector 尚未装配",
        )

    current = await store.get_fish_target(fish_id)
    provided_key = (
        body.api_key.get_secret_value().strip() if body.api_key is not None else ""
    )
    fish_credential_ref = fish_agent_credential_ref(factory, fish_id, body.url)
    if provided_key:
        credential_ref = fish_credential_ref
    elif current is not None and _same_agent_origin(current, candidate_for_origin):
        credential_ref = current.target.credential_ref
    elif _can_inherit_global_credential(request, candidate_for_origin):
        credential_ref = _global_credential_ref(request)
    else:
        credential_ref = None
    if not credential_ref:
        return _target_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "agent_api_key_required",
            "连接其他 Agent 地址时必须填写 API Key",
        )

    target_key = fish_agent_target_key(
        factory,
        fish_id,
        body.url,
        credential_ref,
    )
    configured = factory.target_from_url(
        body.url,
        credential_ref=credential_ref,
        external_key=target_key,
        display_name="Hermes",
    )
    credentials = cast(
        AgentCredentialResolver,
        request.app.state.agent_credential_resolver,
    )
    update_ref = credential_ref
    probe_ref = f"probe:{uuid4().hex}" if provided_key else credential_ref
    probe_target = (
        factory.target_from_url(
            body.url,
            credential_ref=probe_ref,
            external_key=target_key,
            display_name="Hermes",
        )
        if provided_key
        else configured
    )
    async with credentials.update_lock(update_ref):
        previous_key = (
            credentials.resolve_optional(credential_ref) if provided_key else None
        )
        if provided_key:
            credentials.stage(probe_ref, provided_key)
        live_key_changed = False
        saved_configuration = False
        try:
            try:
                capabilities = await connector.probe(probe_target)
            except AgentConnectorError as error:
                return _target_connector_error(error, probing=True)
            if not capabilities.run_submission or not capabilities.run_status:
                return _target_error(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "unsupported_capability",
                    "Agent Endpoint 不支持 Runs 提交和状态查询",
                )
            if provided_key:
                credentials.store(credential_ref, provided_key)
                live_key_changed = True
            saved = await store.upsert_target(
                str(configured.endpoint.connector_type),
                configured.endpoint.base_url,
                configured.target.external_key,
                configured.target.route_prefix,
                configured.target.credential_ref,
                dict(configured.target.connector_config or {}),
                configured.target.display_name,
            )
            if not await store.set_fish_target(fish_id, saved.target.id):
                return _target_error(
                    status.HTTP_404_NOT_FOUND,
                    "fish_not_found",
                    "Fish 不存在",
                )
            saved_configuration = True
            return _fish_agent_configuration_response(request, saved)
        finally:
            if provided_key:
                credentials.unstage(probe_ref)
            if live_key_changed and not saved_configuration:
                if previous_key is None:
                    credentials.remove(credential_ref)
                else:
                    credentials.store(credential_ref, previous_key)


@agent_router.get("/events", response_model=None)
async def stream_agent_messages(
    fish_id: UUID,
    request: Request,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    after_sequence: Annotated[
        int,
        Query(alias="afterSequence", ge=0, le=_MAX_SEQUENCE),
    ] = 0,
) -> StreamingResponse:
    await _require_owned_agent_session(request, fish_id, session_id)
    store = cast(TaskStore, request.app.state.agent_task_store)
    client_cursor = max(after_sequence, _last_event_sequence(request))

    async def events() -> AsyncIterator[str]:
        new_cursor = client_cursor
        sent_out_of_order: dict[UUID, int] = {}
        last_sent = time.monotonic()
        next_incomplete_scan_at = 0.0
        while not await request.is_disconnected():
            emitted = False
            progressed = False
            now = time.monotonic()
            if now >= next_incomplete_scan_at:
                current_incomplete_ids: set[UUID] = set()
                incomplete_scan_sequence = 0
                through_sequence = new_cursor
                while not await request.is_disconnected():
                    items = await store.list_incomplete_agent_messages_for_delivery(
                        fish_id,
                        session_id,
                        after_sequence=incomplete_scan_sequence,
                        through_sequence=through_sequence,
                        limit=_DELIVERY_PAGE_SIZE,
                    )
                    if not items:
                        break
                    progressed = True
                    for item in items:
                        if await request.is_disconnected():
                            return
                        incomplete_scan_sequence = item.sequence
                        current_incomplete_ids.add(item.message_id)
                        if item.message_id in sent_out_of_order:
                            continue
                        sent_out_of_order[item.message_id] = item.sequence
                        emitted = True
                        last_sent = time.monotonic()
                        yield _format_agent_message(item)
                    if len(items) < _DELIVERY_PAGE_SIZE:
                        break

                sent_out_of_order = {
                    message_id: sequence
                    for message_id, sequence in sent_out_of_order.items()
                    if message_id in current_incomplete_ids or sequence > new_cursor
                }
                next_incomplete_scan_at = now + _INCOMPLETE_SCAN_INTERVAL_SECONDS
            if await request.is_disconnected():
                return

            new_pages = 0
            while (
                new_pages < _MAX_NEW_PAGES_PER_CYCLE
                and not await request.is_disconnected()
            ):
                new_items = await store.list_agent_messages_for_delivery(
                    fish_id,
                    session_id,
                    after_sequence=new_cursor,
                    include_incomplete=False,
                    limit=_DELIVERY_PAGE_SIZE,
                    incomplete_after_sequence=0,
                )
                if not new_items:
                    break
                progressed = True
                new_pages += 1
                for item in new_items:
                    if await request.is_disconnected():
                        return
                    new_cursor = item.sequence
                    if item.message_id in sent_out_of_order:
                        continue
                    sent_out_of_order[item.message_id] = item.sequence
                    emitted = True
                    last_sent = time.monotonic()
                    yield _format_agent_message(item)
                if len(new_items) < _DELIVERY_PAGE_SIZE:
                    break

            if not emitted and time.monotonic() - last_sent >= _SSE_HEARTBEAT_SECONDS:
                last_sent = time.monotonic()
                yield "event: agent.heartbeat\ndata: {}\n\n"
            if not emitted and not progressed:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agent_router.post(
    "/{message_id}/consumption",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def consume_agent_message(
    fish_id: UUID,
    message_id: UUID,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    body: AgentMessageConsumptionRequest,
    request: Request,
) -> Response:
    await _require_owned_agent_session(request, fish_id, session_id)
    store = cast(TaskStore, request.app.state.agent_task_store)
    try:
        message = await store.finalize_agent_message_consumption(
            message_id,
            fish_id,
            session_id,
            body.consumed_offset,
            body.interrupted,
        )
    except AgentMessageSubjectConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="message is not a consumable Agent message",
        ) from error
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent message not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _require_owned_agent_session(
    request: Request,
    fish_id: UUID,
    session_id: UUID,
) -> None:
    store = cast(SessionStore, request.app.state.session_store)
    try:
        await store.require_owned(fish_id, session_id)
    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session 不存在或不属于当前 fish",
        ) from error


def _last_event_sequence(request: Request) -> int:
    value = request.headers.get("last-event-id")
    if value is None or _NON_NEGATIVE_INTEGER_RE.fullmatch(value) is None:
        return 0
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed <= _MAX_SEQUENCE else 0


def _format_agent_message(item: AgentMessageEnvelope) -> str:
    data = json.dumps(
        {
            "messageId": str(item.message_id),
            "sequence": item.sequence,
            "sessionId": str(item.session_id),
            "content": item.content,
            "consumedOffset": item.consumed_offset,
            "subject": {
                "kind": item.subject.kind.value,
                "id": str(item.subject.id),
            },
            "activationPolicy": item.activation_policy.value,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {item.sequence}\nevent: agent.message\ndata: {data}\n\n"


async def _parse_fish_agent_configuration(
    request: Request,
) -> FishAgentConfigurationRequest | JSONResponse:
    try:
        return FishAgentConfigurationRequest.model_validate(await request.json())
    except (TypeError, ValueError):
        return _target_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_agent_configuration",
            "Fish Agent 配置请求无效",
        )


def _fish_agent_configuration_response(
    request: Request,
    override: AgentResolvedTarget | None,
) -> FishAgentConfigurationResponse:
    global_url = cast(str | None, getattr(request.app.state, "agent_global_url", None))
    global_configured = bool(
        getattr(request.app.state, "agent_global_configured", False)
    )
    effective_url = global_url if override is None else _resolved_target_url(override)
    return FishAgentConfigurationResponse(
        agent_tasks_enabled=bool(
            getattr(request.app.state.settings, "agent_tasks_enabled", False)
        ),
        configuration_writable=bool(
            getattr(request.app.state, "agent_configuration_writable", False)
        ),
        global_configured=global_configured,
        global_url=global_url,
        inherits_global=override is None,
        url=effective_url,
        has_api_key_override=(
            override is not None
            and override.target.credential_ref != _global_credential_ref(request)
        ),
    )


def _resolved_target_url(target: AgentResolvedTarget) -> str:
    return f"{target.endpoint.base_url.rstrip('/')}{target.target.route_prefix}"


def _same_agent_origin(
    left: AgentResolvedTarget,
    right: AgentResolvedTarget,
) -> bool:
    return _url_origin(left.endpoint.base_url) == _url_origin(right.endpoint.base_url)


def _can_inherit_global_credential(
    request: Request,
    candidate: AgentResolvedTarget,
) -> bool:
    if not bool(getattr(request.app.state, "agent_global_configured", False)):
        return False
    global_url = cast(str | None, getattr(request.app.state, "agent_global_url", None))
    if global_url is None:
        return False
    return _url_origin(global_url) == _url_origin(candidate.endpoint.base_url)


def _global_credential_ref(request: Request) -> str | None:
    return cast(
        str | None,
        getattr(request.app.state, "agent_global_credential_ref", None),
    )


def _url_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


def _target_connector_error(
    error: AgentConnectorError,
    *,
    probing: bool,
) -> JSONResponse:
    deterministic_codes = {
        "authentication_required",
        "permission_denied",
        "invalid_request",
        "unsupported_capability",
    }
    status_code = (
        status.HTTP_422_UNPROCESSABLE_CONTENT
        if not probing or error.code in deterministic_codes
        else status.HTTP_502_BAD_GATEWAY
    )
    return _target_error(
        status_code,
        error.code,
        "Agent 连接校验或能力探测失败",
    )


def _target_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


__all__ = [
    "AgentMessageConsumptionRequest",
    "FishAgentConfigurationRequest",
    "FishAgentConfigurationResponse",
    "agent_router",
    "agent_target_router",
    "consume_agent_message",
    "get_fish_agent_configuration",
    "put_fish_agent_configuration",
    "stream_agent_messages",
]
