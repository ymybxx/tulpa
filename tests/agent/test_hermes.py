from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.agent.connector import (
    AgentConnectorError,
    AgentConnectorRuntime,
    AgentHistoryMessage,
    AgentProviderEventKind,
    AgentRunRequest,
    ConnectorRunState,
)
from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.hermes import HermesConnector, HermesConnectorFactory
from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    AgentTask,
    ConnectorTypeSlug,
)
from fish_memory.db.json_values import canonical_json_object

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 8, 30, tzinfo=UTC)
_FISH_ID = UUID("00000000-0000-0000-0000-000000000111")
_TARGET_ID = UUID("00000000-0000-0000-0000-000000000222")
_LONG_CREDENTIAL = "credential-fragment-" * 2_000


def _resolved_target(
    *,
    base_url: str = "http://localhost:9000/",
    route_prefix: str = "",
    credential_ref: str | None = " hermes:test ",
    connector_config: dict[str, object] | None = None,
) -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID("00000000-0000-0000-0000-000000000333"),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url=base_url,
        created_at=_NOW,
        updated_at=_NOW,
    )
    target = AgentTarget(
        id=_TARGET_ID,
        endpoint_id=endpoint.id,
        external_key="coder",
        route_prefix=route_prefix,
        credential_ref=credential_ref,
        connector_config=connector_config,
        display_name="Hermes Coder",
        enabled=True,
        created_at=_NOW,
        updated_at=_NOW,
    )
    return AgentResolvedTarget(target=target, endpoint=endpoint)


def _work(target: AgentResolvedTarget) -> AgentRunWorkItem:
    task = AgentTask(
        id=UUID("00000000-0000-0000-0000-000000000444"),
        fish_id=_FISH_ID,
        session_id=UUID("00000000-0000-0000-0000-000000000333"),
        target_id=target.target.id,
        external_session_id="fm-task-session-1",
        title="Continue connector implementation",
        created_at=_NOW,
        updated_at=_NOW,
    )
    run = AgentRun(
        id=UUID("00000000-0000-0000-0000-000000000555"),
        task_id=task.id,
        target_id=target.target.id,
        fish_id=task.fish_id,
        session_id=task.session_id,
        previous_run_id=None,
        source_message_id=UUID("00000000-0000-0000-0000-000000000666"),
        confirmation_message_id=None,
        input="Implement the connector",
        external_run_id=None,
        history_source=AgentHistorySource.NONE,
        status=AgentRunStatus.DISPATCHING,
        output=None,
        result_message_id=None,
        error_code=None,
        error_message=None,
        provider_data=None,
        created_at=_NOW,
        started_at=None,
        updated_at=_NOW,
        finished_at=None,
    )
    return AgentRunWorkItem(
        run=run,
        task=task,
        target=target.target,
        endpoint=target.endpoint,
    )


def _json_response(
    status_code: int,
    payload: object,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )


def _connector(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    target: AgentResolvedTarget | None = None,
    resolver: AgentCredentialResolver | None = None,
) -> tuple[Any, AgentResolvedTarget, httpx.AsyncClient]:
    factory = HermesConnectorFactory()
    normalized = factory.validate_target(target or _resolved_target())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = AgentConnectorRuntime(
        credential_resolver=resolver
        or AgentCredentialResolver({"hermes:test": SecretStr("private-token")}),
        http_client=client,
        request_timeout_seconds=5.0,
        sse_read_timeout_seconds=35.0,
    )
    return factory.create(runtime), normalized, client


def _exception_chain(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return chain


def _assert_normalized_secret_safe_error(error: AgentConnectorError) -> None:
    chain = _exception_chain(error)
    assert chain == [error]
    assert "private-token" not in repr(chain)
    assert "provider-secret" not in repr(chain)


class _FailingBodyStream(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self._request = request

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        raise httpx.ReadTimeout(
            "provider-secret body read failed",
            request=self._request,
        )
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


class _UnexpectedFailingBodyStream(httpx.AsyncByteStream):
    async def __aiter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider-secret unexpected stream failure")
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


def _deep_error_json() -> bytes:
    return b'{"error":' + b"[" * 1_200 + b"0" + b"]" * 1_200 + b"}"


async def test_factory_normalizes_target_without_mutating_persisted_models() -> None:
    original = _resolved_target(
        base_url="https://agent.example.test/root///",
        route_prefix="/p/coder/",
        connector_config=None,
    )

    normalized = HermesConnectorFactory().validate_target(original)

    assert normalized.endpoint.base_url == "https://agent.example.test/root"
    assert normalized.target.route_prefix == "/p/coder"
    assert normalized.target.credential_ref == "hermes:test"
    assert normalized.target.connector_config == {}
    assert original.endpoint.base_url == "https://agent.example.test/root///"
    assert original.target.route_prefix == "/p/coder/"
    assert original.target.credential_ref == " hermes:test "
    assert original.target.connector_config is None


async def test_factory_builds_internal_target_from_one_complete_agent_url() -> None:
    configured = HermesConnectorFactory().target_from_url(
        "http://127.0.0.1:8642/p/coder/",
        credential_ref="fish-memory:global",
        external_key="fish-memory:global",
        display_name="Hermes",
    )

    assert configured.endpoint.base_url == "http://127.0.0.1:8642"
    assert configured.target.route_prefix == "/p/coder"
    assert configured.target.credential_ref == "fish-memory:global"
    assert configured.target.external_key == "fish-memory:global"


async def test_factory_builds_default_endpoint_without_a_profile_name() -> None:
    configured = HermesConnectorFactory().target_from_url(
        "http://127.0.0.1:8642/",
        credential_ref="fish-memory:global",
        external_key="fish-memory:global",
        display_name="Hermes",
    )

    assert configured.endpoint.base_url == "http://127.0.0.1:8642"
    assert configured.target.route_prefix == ""


async def test_run_preflight_resolves_credential_without_network() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json_response(500, {"error": "must not be called"})

    connector, target, client = _connector(
        handler,
        target=_resolved_target(credential_ref="hermes:missing"),
        resolver=AgentCredentialResolver({}),
    )
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            connector.preflight_run(_work(target))
    finally:
        await client.aclose()

    assert exc_info.value.code == "authentication_required"
    assert calls == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://agent.example.test",
        "https://user:password@agent.example.test",
        "https://agent.example.test?profile=coder",
        "https://agent.example.test#profile",
        "ftp://agent.example.test",
        "https://",
    ],
)
async def test_factory_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(AgentConnectorError) as exc_info:
        HermesConnectorFactory().validate_target(_resolved_target(base_url=base_url))

    assert exc_info.value.code == "invalid_request"


async def test_factory_suppresses_url_parser_exception_chain() -> None:
    with pytest.raises(AgentConnectorError) as exc_info:
        HermesConnectorFactory().validate_target(
            _resolved_target(base_url="https://agent.example.test:bad-port")
        )

    _assert_normalized_secret_safe_error(exc_info.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "http://[::1]:9000",
        "https://agent.example.test",
    ],
)
async def test_factory_accepts_loopback_http_and_remote_https(base_url: str) -> None:
    normalized = HermesConnectorFactory().validate_target(
        _resolved_target(base_url=base_url)
    )

    assert normalized.endpoint.base_url.startswith(("http://", "https://"))


@pytest.mark.parametrize(
    "route_prefix",
    ["p/coder", "/p//coder", "/p/../coder", "/p/coder?debug=1", "/p/coder#x"],
)
async def test_factory_rejects_malformed_route_prefix(route_prefix: str) -> None:
    with pytest.raises(AgentConnectorError) as exc_info:
        HermesConnectorFactory().validate_target(
            _resolved_target(route_prefix=route_prefix)
        )

    assert exc_info.value.code == "invalid_request"


async def test_start_run_uses_exact_default_route_headers_and_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(202, {"run_id": "run_123", "status": "started"})

    connector, target, client = _connector(handler)
    try:
        accepted = await connector.start_run(
            _work(target),
            AgentRunRequest(input="Do the work", instructions="Be precise"),
        )
    finally:
        await connector.close()
        await client.aclose()

    assert accepted.external_run_id == "run_123"
    assert accepted.state is ConnectorRunState.RUNNING
    assert len(seen) == 1
    request = seen[0]
    assert request.url.path == "/v1/runs"
    assert request.headers["Authorization"] == "Bearer private-token"
    assert (
        request.headers["X-Hermes-Session-Key"]
        == "fm-fish:00000000-0000-0000-0000-000000000111:"
        "target:00000000-0000-0000-0000-000000000222"
    )
    assert json.loads(request.content) == {
        "input": "Do the work",
        "session_id": "fm-task-session-1",
        "instructions": "Be precise",
    }


async def test_start_run_preserves_valid_id_when_provider_status_is_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(202, {"run_id": "run_unknown", "status": "new-phase"})

    connector, target, client = _connector(handler)
    try:
        accepted = await connector.start_run(
            _work(target),
            AgentRunRequest(input="Do the work"),
        )
    finally:
        await connector.close()
        await client.aclose()

    assert accepted.external_run_id == "run_unknown"
    assert accepted.state is ConnectorRunState.UNKNOWN


@pytest.mark.parametrize(
    "provider_run_id",
    ["r" * 513, "run-private-token-reflection"],
)
async def test_start_run_rejects_unpersistable_or_secret_run_id_as_uncertain(
    provider_run_id: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            202,
            {"run_id": provider_run_id, "status": "started"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(
                _work(target),
                AgentRunRequest(input="Do the work"),
            )
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "dispatch_unknown"
    assert exc_info.value.dispatch_uncertain is True
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_long_credential_run_id_is_rejected_without_prefix_reflection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            202,
            {"run_id": _LONG_CREDENTIAL, "status": "started"},
        )

    connector, target, client = _connector(
        handler,
        resolver=AgentCredentialResolver({"hermes:test": SecretStr(_LONG_CREDENTIAL)}),
    )
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(
                _work(target),
                AgentRunRequest(input="Do the work"),
            )
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "dispatch_unknown"
    assert exc_info.value.dispatch_uncertain is True
    assert _LONG_CREDENTIAL[:512] not in repr(_exception_chain(exc_info.value))


async def test_start_run_uses_profile_route_and_explicit_bounded_history() -> None:
    seen_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/p/coder/v1/runs"
        seen_body.update(json.loads(request.content))
        return _json_response(202, {"run_id": "run_history", "status": "queued"})

    connector, target, client = _connector(
        handler,
        target=_resolved_target(route_prefix="/p/coder"),
    )
    try:
        await connector.start_run(
            _work(target),
            AgentRunRequest(
                input="Continue",
                conversation_history=(
                    AgentHistoryMessage(role="user", content="Earlier request"),
                    AgentHistoryMessage(role="assistant", content="Earlier answer"),
                ),
            ),
        )
    finally:
        await connector.close()
        await client.aclose()

    assert seen_body == {
        "input": "Continue",
        "session_id": "fm-task-session-1",
        "conversation_history": [
            {"role": "user", "content": "Earlier request"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
    }
    assert "model" not in seen_body
    assert "provider" not in seen_body
    assert "previous_response_id" not in seen_body


async def test_start_run_omits_empty_optional_instructions() -> None:
    seen_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return _json_response(202, {"run_id": "run_empty", "status": "started"})

    connector, target, client = _connector(handler)
    try:
        await connector.start_run(
            _work(target),
            AgentRunRequest(input="Do the work", instructions=""),
        )
    finally:
        await connector.close()
        await client.aclose()

    assert "instructions" not in seen_body


async def test_start_run_rejects_unbounded_history_before_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json_response(202, {"run_id": "unexpected", "status": "started"})

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(
                _work(target),
                AgentRunRequest(
                    input="Continue",
                    conversation_history=(
                        AgentHistoryMessage(role="user", content="x" * 8_001),
                    ),
                ),
            )
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 0
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.dispatch_uncertain is False


async def test_start_run_rejects_oversized_json_before_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json_response(202, {"run_id": "unexpected", "status": "started"})

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(
                _work(target),
                AgentRunRequest(input="x" * 1_048_576),
            )
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 0
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.dispatch_uncertain is False


async def test_every_request_resolves_the_current_target_credential() -> None:
    calls = 0

    class RotatingResolver(AgentCredentialResolver):
        def resolve(self, credential_ref: str) -> str:
            nonlocal calls
            calls += 1
            return f"token-{calls}"

    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["Authorization"])
        if request.url.path.endswith("/capabilities"):
            return _json_response(200, {"features": {}})
        return _json_response(200, {"status": "running", "run_id": "run_1"})

    connector, target, client = _connector(handler, resolver=RotatingResolver({}))
    try:
        await connector.probe(target)
        await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert authorizations == ["Bearer token-1", "Bearer token-2"]


async def test_probe_maps_only_supported_hermes_capabilities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/capabilities"
        return _json_response(
            200,
            {
                "object": "hermes.api_server.capabilities",
                "platform": "hermes-agent",
                "features": {
                    "run_submission": True,
                    "run_status": True,
                    "run_events_sse": True,
                    "run_stop": True,
                    "run_steer": True,
                    "run_approval_response": True,
                    "session_resources": True,
                    "private_future_feature": True,
                },
            },
        )

    connector, target, client = _connector(handler)
    try:
        capabilities = await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    assert capabilities.run_submission is True
    assert capabilities.run_status is True
    assert capabilities.run_events is True
    assert capabilities.session_history is True
    assert capabilities.approval is True
    assert capabilities.steer is True
    assert capabilities.stop is True


async def test_session_history_uses_exact_latest_query_and_safe_projection() -> None:
    seen_url: httpx.URL | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = request.url
        return _json_response(
            200,
            {
                "object": "list",
                "session_id": "fm-task-session-1",
                "data": [
                    {"role": "user", "content": "first"},
                    {"role": "tool", "content": "hidden tool output"},
                    {"role": "assistant", "content": {"type": "text"}},
                    {"role": "assistant", "content": "second"},
                ],
                "pagination": {"order": "latest", "limit": 64, "returned": 4},
            },
        )

    connector, target, client = _connector(handler)
    try:
        history = await connector.get_session_history(_work(target))
    finally:
        await connector.close()
        await client.aclose()

    assert seen_url is not None
    assert seen_url.path == "/api/sessions/fm-task-session-1/messages"
    assert seen_url.query == b"order=latest&limit=64"
    assert history == (
        AgentHistoryMessage(role="user", content="first"),
        AgentHistoryMessage(role="assistant", content="second"),
    )


async def test_session_history_enforces_single_character_and_total_byte_bounds() -> (
    None
):
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 8_500}
        for index in range(64)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"object": "list", "data": messages})

    connector, target, client = _connector(handler)
    try:
        history = await connector.get_session_history(_work(target))
    finally:
        await connector.close()
        await client.aclose()

    assert all(len(message.content) <= 8_000 for message in history)
    assert sum(len(message.content.encode()) for message in history) <= 256_000
    assert history[-1].content == "x" * 8_000


@pytest.mark.parametrize(
    ("provider_status", "expected"),
    [
        ("queued", ConnectorRunState.RUNNING),
        ("started", ConnectorRunState.RUNNING),
        ("running", ConnectorRunState.RUNNING),
        ("waiting_for_approval", ConnectorRunState.WAITING_USER),
        ("stopping", ConnectorRunState.RUNNING),
        ("completed", ConnectorRunState.COMPLETED),
        ("failed", ConnectorRunState.FAILED),
        ("cancelled", ConnectorRunState.CANCELLED),
    ],
)
async def test_run_snapshot_normalizes_closed_provider_states(
    provider_status: str,
    expected: ConnectorRunState,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "object": "hermes.run",
                "run_id": "run_state",
                "status": provider_status,
                "output": "authoritative result",
                "error": "bounded failure",
                "pending_steer": "Please also verify lint",
                "usage": {"total_tokens": 123},
                "model": "provider-private-model",
            },
        )

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_state")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is expected
    assert snapshot.output == "authoritative result"
    assert snapshot.error_message == "bounded failure"
    expected_provider_data: dict[str, object] = {
        "pending_steer": "Please also verify lint",
        "usage": {"total_tokens": 123},
    }
    if provider_status == "stopping":
        expected_provider_data["provider_status"] = "stopping"
    assert snapshot.provider_data == expected_provider_data
    assert "provider-private-model" not in repr(snapshot.provider_data)


async def test_unknown_snapshot_status_is_reconcilable_provider_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"run_id": "run_1", "status": "teleported"})

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is ConnectorRunState.UNKNOWN
    assert snapshot.provider_data == {"provider_status": "teleported"}


async def test_unknown_snapshot_status_is_bounded_and_secret_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {"run_id": "run_1", "status": "private-token-" + "x" * 200},
        )

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is ConnectorRunState.UNKNOWN
    assert snapshot.provider_data is not None
    provider_status = snapshot.provider_data["provider_status"]
    assert isinstance(provider_status, str)
    assert len(provider_status) <= 128
    assert "private-token" not in provider_status


@pytest.mark.parametrize(
    ("provider_status", "expected_state", "expected_provider_data"),
    [
        ("running", ConnectorRunState.RUNNING, None),
        ("failed", ConnectorRunState.FAILED, None),
        (
            "stopping",
            ConnectorRunState.RUNNING,
            {"provider_status": "[REDACTED]"},
        ),
    ],
)
async def test_status_mapping_uses_raw_value_before_secret_safe_metadata_copy(
    provider_status: str,
    expected_state: ConnectorRunState,
    expected_provider_data: dict[str, object] | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {"run_id": "run_1", "status": provider_status},
        )

    connector, target, client = _connector(
        handler,
        resolver=AgentCredentialResolver({"hermes:test": SecretStr(provider_status)}),
    )
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is expected_state
    assert snapshot.provider_data == expected_provider_data
    if expected_provider_data is not None:
        assert provider_status not in repr(snapshot.provider_data)


async def test_snapshot_redacts_exact_resolved_credential_from_safe_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "run_id": "run_1",
                "status": "failed",
                "error": "private-token failed",
                "pending_steer": "retry with private-token",
            },
        )

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.error_message == "[REDACTED] failed"
    assert snapshot.provider_data == {"pending_steer": "retry with [REDACTED]"}


async def test_snapshot_bounds_and_redacts_terminal_output_for_persistence() -> None:
    reflected_output = (
        "private-token Bearer provider-secret api_key=provider-secret " + "x" * 40_000
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "run_id": "run_1",
                "status": "completed",
                "output": reflected_output,
            },
        )

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.output is not None
    assert len(snapshot.output) <= 32_000
    assert "private-token" not in snapshot.output
    assert "provider-secret" not in snapshot.output
    assert snapshot.provider_data == {"output_truncated": True}


async def test_long_credential_is_redacted_before_every_snapshot_field_bound() -> None:
    bearer_token = "bearer-fragment-" * 2_000
    assignment_token = "assignment-fragment-" * 2_000
    reflected_output = (
        _LONG_CREDENTIAL + " Bearer " + bearer_token + " api_key=" + assignment_token
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "run_id": "run_1",
                "status": _LONG_CREDENTIAL,
                "output": reflected_output,
                "error": _LONG_CREDENTIAL,
                "pending_steer": _LONG_CREDENTIAL,
            },
        )

    connector, target, client = _connector(
        handler,
        resolver=AgentCredentialResolver({"hermes:test": SecretStr(_LONG_CREDENTIAL)}),
    )
    try:
        snapshot = await connector.get_run(_work(target), "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is ConnectorRunState.UNKNOWN
    assert snapshot.output is not None
    assert len(snapshot.output) <= 32_000
    assert snapshot.error_message == "[REDACTED]"
    assert snapshot.provider_data is not None
    assert snapshot.provider_data["pending_steer"] == "[REDACTED]"
    assert snapshot.provider_data["provider_status"] == "[REDACTED]"
    outward = repr(snapshot)
    assert _LONG_CREDENTIAL[:8_000] not in outward
    assert bearer_token[:8_000] not in outward
    assert assignment_token[:8_000] not in outward


async def test_missing_external_run_snapshot_is_unknown_not_dispatchable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"error": {"code": "run_not_found"}})

    connector, target, client = _connector(handler)
    try:
        snapshot = await connector.get_run(_work(target), "run_missing")
    finally:
        await connector.close()
        await client.aclose()

    assert snapshot.state is ConnectorRunState.UNKNOWN


async def test_approval_steer_and_stop_use_exact_paths_and_payloads() -> None:
    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.content))
        return _json_response(200, {"ok": True})

    connector, target, client = _connector(handler)
    work = _work(target)
    running_work = replace(work, run=replace(work.run, status=AgentRunStatus.RUNNING))
    try:
        await connector.respond_approval(work, "run_1", "session", True)
        await connector.steer_run(running_work, "run_1", "Focus on tests")
        await connector.stop_run(work, "run_1")
    finally:
        await connector.close()
        await client.aclose()

    assert seen == [
        (
            "/v1/runs/run_1/approval",
            b'{"choice":"session","resolve_all":true}',
        ),
        ("/v1/runs/run_1/steer", b'{"input":"Focus on tests"}'),
        ("/v1/runs/run_1/stop", b""),
    ]


async def test_steer_rejects_non_running_local_run_before_http() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json_response(200, {"accepted": True})

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.steer_run(
                _work(target),
                "run_1",
                "Focus on tests",
            )
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 0
    assert exc_info.value.code == "invalid_request"


async def test_approval_rejects_choices_outside_the_hermes_contract() -> None:
    connector, target, client = _connector(
        lambda request: _json_response(200, {"ok": True})
    )
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.respond_approval(_work(target), "run_1", "approve", False)
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "invalid_request"


async def test_sse_ignores_private_payloads_and_normalizes_approval() -> None:
    sse = b"".join(
        [
            b": keepalive\n\n",
            b"event: message.delta\n"
            b'data: {"event":"message.delta","delta":"secret"}\n\n',
            b'data: {"event":"tool.started","preview":"private"}\n\n',
            b"event: approval.request\n",
            b'data: {"event":"approval.request",\n',
            b'data: "command":"deploy redacted",\n',
            b'data: "choices":["once","deny"]}\n\n',
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(
            200, content=sse, headers={"Content-Type": "text/event-stream"}
        )

    connector, target, client = _connector(handler)
    try:
        events = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert len(events) == 1
    event = events[0]
    assert event.kind is AgentProviderEventKind.APPROVAL
    assert event.state is ConnectorRunState.WAITING_USER
    assert event.approval_choices == ("once", "deny")
    assert event.provider_data == {"command": "deploy redacted"}


async def test_long_credentials_are_redacted_before_sse_approval_field_bounds() -> None:
    payload = {
        "event": "approval.request",
        "choices": ["once", "deny"],
        "command": _LONG_CREDENTIAL,
        "tool": "Bearer " + _LONG_CREDENTIAL,
        "description": "api_key=" + _LONG_CREDENTIAL,
        "reason": _LONG_CREDENTIAL,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: " + json.dumps(payload).encode() + b"\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(
        handler,
        resolver=AgentCredentialResolver({"hermes:test": SecretStr(_LONG_CREDENTIAL)}),
    )
    try:
        events = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert len(events) == 1
    assert events[0].provider_data == {
        "command": "[REDACTED]",
        "tool": "Bearer [REDACTED]",
        "description": "api_key=[REDACTED]",
        "reason": "[REDACTED]",
    }
    assert _LONG_CREDENTIAL[:4_000] not in repr(events[0])


async def test_approval_provider_data_fits_task_store_canonical_json_limit() -> None:
    worst_case = "\x01" * 4_000
    payload = {
        "event": "approval.request",
        "choices": ["once", "deny"],
        "command": worst_case,
        "tool": worst_case,
        "description": worst_case,
        "reason": worst_case,
        "smart_denied": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: " + json.dumps(payload).encode() + b"\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    try:
        events = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    provider_data = events[0].provider_data
    assert provider_data is not None
    assert tuple(provider_data)[:2] == ("command", "tool")
    serialized = canonical_json_object(
        provider_data,
        label="approval provider_data",
        maximum_bytes=65_536,
    )
    assert serialized is not None
    assert len(serialized.encode("utf-8")) <= 65_536
    assert all(
        forbidden not in provider_data
        for forbidden in ("secret", "token", "authorization", "api_key")
    )


async def test_sse_stopping_signal_stays_core_running_with_provider_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"event":"run.stopping"}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    try:
        events = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert len(events) == 1
    assert events[0].state is ConnectorRunState.RUNNING
    assert events[0].provider_data == {"provider_status": "stopping"}


async def test_terminal_sse_is_closed_with_authoritative_snapshot() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                content=b'data: {"event":"run.completed","output":"stale"}\n\n',
                headers={"Content-Type": "text/event-stream"},
            )
        return _json_response(
            200,
            {"run_id": "run_1", "status": "completed", "output": "authoritative"},
        )

    connector, target, client = _connector(handler)
    try:
        events = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert paths == ["/v1/runs/run_1/events", "/v1/runs/run_1"]
    assert len(events) == 1
    assert events[0].kind is AgentProviderEventKind.TERMINAL
    assert events[0].state is ConnectorRunState.COMPLETED
    assert events[0].output == "authoritative"


async def test_duplicate_concurrent_watchers_fail_closed() -> None:
    release = asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b": keepalive\n\n"
            await release.wait()

        async def aclose(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=WaitingStream(),
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    first = connector.watch_run(_work(target), "run_1")
    try:
        first_next = asyncio.create_task(anext(first))
        await asyncio.sleep(0)
        with pytest.raises(AgentConnectorError) as exc_info:
            await anext(connector.watch_run(_work(target), "run_1"))
        assert exc_info.value.code == "invalid_request"
    finally:
        release.set()
        first_next.cancel()
        await connector.close()
        await client.aclose()


async def test_same_provider_run_id_can_be_watched_on_two_targets() -> None:
    release = asyncio.Event()
    calls = 0

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b": keepalive\n\n"
            await release.wait()

        async def aclose(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=WaitingStream(),
            headers={"Content-Type": "text/event-stream"},
        )

    connector, first_target, client = _connector(handler)
    second_target = replace(
        first_target,
        target=replace(
            first_target.target,
            id=UUID("00000000-0000-0000-0000-000000000777"),
        ),
    )
    pending = [
        asyncio.create_task(
            anext(connector.watch_run(_work(target), "shared-provider-run"))
        )
        for target in (first_target, second_target)
    ]
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls == 2
    finally:
        release.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await connector.close()
        await client.aclose()


@pytest.mark.parametrize(
    ("status_code", "expected_code", "retryable"),
    [
        (400, "invalid_request", False),
        (401, "authentication_required", False),
        (403, "permission_denied", False),
        (404, "invalid_request", False),
        (409, "invalid_request", False),
        (500, "provider_error", True),
        (503, "provider_error", True),
    ],
)
async def test_http_errors_are_normalized_and_secret_safe(
    status_code: int,
    expected_code: str,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            status_code,
            {"error": {"message": "Bearer private-token failed for api_key=hidden"}},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    error = exc_info.value
    assert error.code == expected_code
    assert error.retryable is retryable
    assert error.dispatch_uncertain is False
    assert len(str(error)) <= 4_000
    assert "private-token" not in str(error)
    assert "hidden" not in str(error)


async def test_rate_limit_is_definitive_and_preserves_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            429,
            {"error": {"message": "slow down"}},
            headers={"Retry-After": "12"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "rate_limited"
    assert exc_info.value.retry_after == "12"
    assert exc_info.value.dispatch_uncertain is False


async def test_retry_after_is_bounded_and_secret_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            429,
            {"error": {"message": "slow down"}},
            headers={"Retry-After": "private-token " + "x" * 200},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    retry_after = exc_info.value.retry_after
    assert retry_after is not None
    assert len(retry_after) <= 128
    assert "private-token" not in retry_after


async def test_long_credential_retry_after_is_redacted_before_bound() -> None:
    response = httpx.Response(
        429,
        content=b'{"error":{"message":"slow down"}}',
        headers={"Retry-After": _LONG_CREDENTIAL},
    )
    connector, target, client = _connector(
        lambda request: response,
        resolver=AgentCredentialResolver({"hermes:test": SecretStr(_LONG_CREDENTIAL)}),
    )
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.retry_after == "[REDACTED]"
    assert _LONG_CREDENTIAL[:128] not in repr(_exception_chain(exc_info.value))


@pytest.mark.parametrize(
    ("status_code", "expected_code", "dispatch_uncertain"),
    [
        (307, "protocol_error", False),
        (400, "invalid_request", False),
        (401, "authentication_required", False),
        (403, "permission_denied", False),
        (404, "invalid_request", False),
        (409, "invalid_request", False),
        (429, "rate_limited", False),
        (500, "provider_error", True),
    ],
)
async def test_http_status_classification_survives_error_body_read_failure(
    status_code: int,
    expected_code: str,
    dispatch_uncertain: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": "5"} if status_code == 429 else None
        return httpx.Response(
            status_code,
            stream=_FailingBodyStream(request),
            headers=headers,
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == expected_code
    assert exc_info.value.dispatch_uncertain is dispatch_uncertain
    _assert_normalized_secret_safe_error(exc_info.value)


@pytest.mark.parametrize(
    ("status_code", "expected_code", "dispatch_uncertain"),
    [
        (307, "protocol_error", False),
        (400, "invalid_request", False),
        (401, "authentication_required", False),
        (403, "permission_denied", False),
        (404, "invalid_request", False),
        (409, "invalid_request", False),
        (429, "rate_limited", False),
        (500, "provider_error", True),
    ],
)
async def test_http_status_classification_survives_deep_error_json(
    status_code: int,
    expected_code: str,
    dispatch_uncertain: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": "5"} if status_code == 429 else None
        return httpx.Response(
            status_code,
            content=_deep_error_json(),
            headers=headers,
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == expected_code
    assert exc_info.value.dispatch_uncertain is dispatch_uncertain
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_malformed_error_json_falls_back_to_status_without_secret_leak() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content=b'{"error":{"message":"private-token',
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.dispatch_uncertain is False
    assert "private-token" not in str(exc_info.value)


async def test_error_body_parser_does_not_catch_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt_parser(value: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr("fish_memory.agent.hermes.json.loads", interrupt_parser)
    response = httpx.Response(400, content=b"{}")
    connector, target, client = _connector(lambda request: response)
    try:
        with pytest.raises(KeyboardInterrupt):
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()


async def test_definitive_status_survives_unexpected_error_body_stream_failure() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, stream=_UnexpectedFailingBodyStream())

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.dispatch_uncertain is False
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_oversized_400_body_remains_a_definitive_client_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, content=b"x" * 1_048_577)

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 1
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.dispatch_uncertain is False
    assert len(str(exc_info.value)) <= 4_000


async def test_redirect_is_protocol_error_and_is_not_followed() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"Location": "https://attacker.test/steal"})

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 1
    assert exc_info.value.code == "protocol_error"


async def test_post_redirect_is_definitive_protocol_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"Location": "https://attacker.test/steal"})

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 1
    assert exc_info.value.code == "protocol_error"
    assert exc_info.value.dispatch_uncertain is False


async def test_post_timeout_is_attempted_once_and_dispatch_is_uncertain() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("raw transport detail private-token", request=request)

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert calls == 1
    assert exc_info.value.code == "timeout"
    assert exc_info.value.retryable is False
    assert exc_info.value.dispatch_uncertain is True
    assert "private-token" not in str(exc_info.value)
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_malformed_json_does_not_retain_provider_decode_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"provider-secret":',
            headers={"Content-Type": "application/json"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "protocol_error"
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_malformed_sse_utf8_does_not_retain_decode_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: \xff-provider-secret\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            _ = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "protocol_error"
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_malformed_sse_json_does_not_retain_parse_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"provider-secret":\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            _ = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "protocol_error"
    _assert_normalized_secret_safe_error(exc_info.value)


async def test_missing_credential_does_not_retain_resolver_exception_chain() -> None:
    connector, target, client = _connector(
        lambda request: _json_response(200, {"features": {}}),
        resolver=AgentCredentialResolver({}),
    )
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "authentication_required"
    _assert_normalized_secret_safe_error(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, content=b"provider unavailable"),
        httpx.Response(202, content=b"not-json"),
        _json_response(202, {"status": "started"}),
    ],
)
async def test_post_ambiguous_responses_are_dispatch_uncertain(
    response: httpx.Response,
) -> None:
    connector, target, client = _connector(lambda request: response)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.start_run(_work(target), AgentRunRequest(input="work"))
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.dispatch_uncertain is True
    assert exc_info.value.retryable is False


async def test_oversized_json_body_fails_before_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'"' + b"x" * 1_048_576 + b'"',
            headers={"Content-Type": "application/json"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            await connector.probe(target)
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "protocol_error"
    assert len(str(exc_info.value)) <= 4_000


async def test_oversized_sse_frame_fails_before_json_parsing() -> None:
    frame = b"data: " + b"x" * 262_144 + b"\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=frame,
            headers={"Content-Type": "text/event-stream"},
        )

    connector, target, client = _connector(handler)
    try:
        with pytest.raises(AgentConnectorError) as exc_info:
            _ = [event async for event in connector.watch_run(_work(target), "run_1")]
    finally:
        await connector.close()
        await client.aclose()

    assert exc_info.value.code == "protocol_error"


async def test_connector_does_not_close_injected_http_client() -> None:
    connector, target, client = _connector(
        lambda request: _json_response(200, {"features": {}})
    )

    await connector.probe(target)
    await connector.close()

    assert client.is_closed is False
    await client.aclose()


async def test_connector_closes_the_http_client_it_creates() -> None:
    connector = HermesConnectorFactory().create(
        AgentConnectorRuntime(
            credential_resolver=AgentCredentialResolver(
                {"hermes:test": SecretStr("private-token")}
            )
        )
    )
    assert isinstance(connector, HermesConnector)
    owned_client = connector._client

    assert owned_client.is_closed is False
    await connector.close()

    assert owned_client.is_closed is True


async def test_timeout_update_applies_to_new_requests_without_canceling_an_active_one():
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    async def handle(request):
        observed.append(dict(request.extensions["timeout"]))
        if len(observed) == 1:
            entered.set()
            await release.wait()
        return httpx.Response(200, json={"features": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        connector = HermesConnector(
            AgentConnectorRuntime(
                credential_resolver=AgentCredentialResolver(
                    {"hermes:test": SecretStr("fixture-key")}
                ),
                http_client=http,
                request_timeout_seconds=15,
                sse_read_timeout_seconds=45,
            )
        )
        target = _resolved_target(credential_ref="hermes:test")
        current = asyncio.create_task(connector.probe(target))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            connector.configure_timeouts(request_seconds=4, sse_read_seconds=20)
            await connector.probe(target)
            assert not current.done()
            release.set()
            await current
            assert [item["read"] for item in observed] == [15, 4]
        finally:
            current.cancel()
            await asyncio.gather(current, return_exceptions=True)
            await connector.close()
