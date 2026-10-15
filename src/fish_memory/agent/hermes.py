from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import UUID, uuid4

import httpx

from fish_memory.agent.connector import (
    AGENT_CONNECTOR_CONTRACT_VERSION,
    AgentCapabilities,
    AgentConnectorError,
    AgentConnectorRuntime,
    AgentHistoryMessage,
    AgentProviderEvent,
    AgentProviderEventKind,
    AgentRunAccepted,
    AgentRunRequest,
    AgentRunSnapshot,
    ConnectorDescriptor,
    ConnectorRunState,
)
from fish_memory.agent.credentials import AgentCredentialLookupError
from fish_memory.agent.models import (
    AgentEndpoint,
    AgentResolvedTarget,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    ConnectorTypeSlug,
)

_MAX_JSON_BODY_BYTES = 1_048_576
_MAX_SSE_FRAME_BYTES = 262_144
_MAX_ERROR_TEXT_CHARS = 4_000
_MAX_EXTERNAL_RUN_ID_CHARS = 512
_MAX_OUTPUT_CHARS = 32_000
_MAX_HISTORY_MESSAGE_CHARS = 8_000
_MAX_HISTORY_BYTES = 256_000
_MAX_PENDING_STEER_CHARS = 8_000
_MAX_PROVIDER_DATA_BYTES = 65_536
_MAX_PROVIDER_STATUS_CHARS = 128
_MAX_RETRY_AFTER_CHARS = 128

_ROUTE_PREFIX_RE = re.compile(r"^(?:/[A-Za-z0-9._~-]+)*$")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;\"']+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;\"']+"
)

_STATE_MAP: dict[str, ConnectorRunState] = {
    "queued": ConnectorRunState.RUNNING,
    "started": ConnectorRunState.RUNNING,
    "running": ConnectorRunState.RUNNING,
    "waiting_for_approval": ConnectorRunState.WAITING_USER,
    "stopping": ConnectorRunState.RUNNING,
    "completed": ConnectorRunState.COMPLETED,
    "failed": ConnectorRunState.FAILED,
    "cancelled": ConnectorRunState.CANCELLED,
}

_APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})
_NO_BODY = object()
_OUTPUT_TRUNCATED_KEY = "_fish_memory_output_truncated"
_SAFE_PROVIDER_STATUS_KEY = "_fish_memory_safe_provider_status"


class HermesConnectorFactory:
    descriptor = ConnectorDescriptor(
        type_slug="hermes",
        contract_version=AGENT_CONNECTOR_CONTRACT_VERSION,
        display_name="Hermes Agent",
    )

    def validate_target(self, target: AgentResolvedTarget) -> AgentResolvedTarget:
        base_url = _normalize_base_url(target.endpoint.base_url)
        route_prefix = _normalize_route_prefix(target.target.route_prefix)
        credential_ref = (target.target.credential_ref or "").strip()
        if not credential_ref:
            raise AgentConnectorError(
                "Hermes target requires a credential reference",
                code="authentication_required",
            )
        connector_config = dict(target.target.connector_config or {})
        normalized_endpoint = replace(target.endpoint, base_url=base_url)
        normalized_target = replace(
            target.target,
            route_prefix=route_prefix,
            credential_ref=credential_ref,
            connector_config=connector_config,
        )
        return AgentResolvedTarget(
            target=normalized_target,
            endpoint=normalized_endpoint,
        )

    def target_from_url(
        self,
        agent_url: str,
        *,
        credential_ref: str,
        external_key: str,
        display_name: str,
    ) -> AgentResolvedTarget:
        normalized_url = _normalize_base_url(agent_url)
        parsed = urlsplit(normalized_url)
        base_url = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        now = datetime.now(UTC)
        endpoint_id = uuid4()
        return self.validate_target(
            AgentResolvedTarget(
                target=AgentTarget(
                    id=uuid4(),
                    endpoint_id=endpoint_id,
                    external_key=external_key,
                    route_prefix=parsed.path,
                    credential_ref=credential_ref,
                    connector_config={},
                    display_name=display_name,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                ),
                endpoint=AgentEndpoint(
                    id=endpoint_id,
                    connector_type=ConnectorTypeSlug(self.descriptor.type_slug),
                    base_url=base_url,
                    created_at=now,
                    updated_at=now,
                ),
            )
        )

    def create(self, runtime: AgentConnectorRuntime) -> HermesConnector:
        return HermesConnector(runtime)


class HermesConnector:
    def __init__(self, runtime: AgentConnectorRuntime) -> None:
        self._credential_resolver = runtime.credential_resolver
        self._request_timeout_seconds = runtime.request_timeout_seconds
        self._sse_read_timeout_seconds = runtime.sse_read_timeout_seconds
        self._owns_client = runtime.http_client is None
        self._client = runtime.http_client or httpx.AsyncClient(follow_redirects=False)
        self._watched_runs: set[tuple[UUID, str]] = set()

    def configure_timeouts(
        self, *, request_seconds: float, sse_read_seconds: float
    ) -> None:
        if request_seconds <= 0 or sse_read_seconds <= 0:
            raise ValueError("Agent timeouts must be positive")
        # httpx snapshots these values when the next request is constructed.
        # An established SSE connection retains its original timeout.
        self._request_timeout_seconds = request_seconds
        self._sse_read_timeout_seconds = sse_read_seconds

    async def probe(self, target: AgentResolvedTarget) -> AgentCapabilities:
        payload = await self._request_json("GET", target, "/v1/capabilities")
        features = payload.get("features")
        if not isinstance(features, Mapping):
            raise _protocol_error("Hermes capabilities response is malformed")
        return AgentCapabilities(
            run_submission=features.get("run_submission") is True,
            run_status=features.get("run_status") is True,
            run_events=features.get("run_events_sse") is True,
            session_history=features.get("session_resources") is True,
            approval=features.get("run_approval_response") is True,
            steer=features.get("run_steer") is True,
            stop=features.get("run_stop") is True,
        )

    def preflight_run(self, work: AgentRunWorkItem) -> AgentRunWorkItem:
        normalized = HermesConnectorFactory().validate_target(_resolved_target(work))
        self._credential(normalized)
        return replace(
            work,
            target=normalized.target,
            endpoint=normalized.endpoint,
        )

    async def start_run(
        self,
        work: AgentRunWorkItem,
        request: AgentRunRequest,
    ) -> AgentRunAccepted:
        if not isinstance(request.input, str) or not request.input.strip():
            raise AgentConnectorError(
                "Agent run input must be non-empty",
                code="invalid_request",
            )
        body: dict[str, object] = {
            "input": request.input,
            "session_id": work.task.external_session_id,
        }
        if request.instructions:
            body["instructions"] = request.instructions
        if request.conversation_history:
            _validate_outbound_history(request.conversation_history)
            body["conversation_history"] = [
                {"role": message.role, "content": message.content}
                for message in request.conversation_history
            ]
        payload = await self._request_json(
            "POST",
            _resolved_target(work),
            "/v1/runs",
            json_body=body,
            dispatch=True,
            session_scope=_session_scope(work),
            expected_status=202,
        )
        run_id = payload.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or len(run_id) > _MAX_EXTERNAL_RUN_ID_CHARS
        ):
            raise AgentConnectorError(
                "Hermes accepted the run without a usable run identifier",
                code="dispatch_unknown",
                retryable=False,
                dispatch_uncertain=True,
            )
        raw_status = payload.get("status")
        if not isinstance(raw_status, str):
            raise AgentConnectorError(
                "Hermes accepted the run with a malformed status",
                code="dispatch_unknown",
                dispatch_uncertain=True,
            )
        state = _STATE_MAP.get(raw_status, ConnectorRunState.UNKNOWN)
        return AgentRunAccepted(external_run_id=run_id, state=state)

    async def watch_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> AsyncIterator[AgentProviderEvent]:
        watcher_key = (work.target.id, external_run_id)
        if watcher_key in self._watched_runs:
            raise AgentConnectorError(
                "A Hermes run can have only one active event watcher",
                code="invalid_request",
            )
        self._watched_runs.add(watcher_key)
        target = _resolved_target(work)
        path = f"/v1/runs/{quote(external_run_id, safe='')}/events"
        try:
            headers, credential = self._headers(target)
            headers["Accept"] = "text/event-stream"
            timeout = httpx.Timeout(
                self._request_timeout_seconds,
                read=self._sse_read_timeout_seconds,
            )
            normalized_error: AgentConnectorError | None = None
            try:
                async with self._client.stream(
                    "GET",
                    self._url(target, path),
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if not 200 <= response.status_code < 300:
                        body = await _read_bounded_error_body(response)
                        raise _http_error(
                            response,
                            body,
                            credential=credential,
                            dispatch=False,
                        )
                    async for payload, event_name in _iter_sse_payloads(response):
                        provider_event = str(event_name or payload.get("event") or "")
                        if provider_event == "approval.request":
                            yield _approval_event(payload, credential=credential)
                            continue
                        if provider_event in {
                            "run.queued",
                            "run.started",
                            "run.running",
                            "approval.responded",
                            "run.stopping",
                        }:
                            yield AgentProviderEvent(
                                kind=AgentProviderEventKind.STATE,
                                state=ConnectorRunState.RUNNING,
                                provider_data=(
                                    {"provider_status": "stopping"}
                                    if provider_event == "run.stopping"
                                    else None
                                ),
                            )
                            continue
                        if provider_event in {
                            "run.completed",
                            "run.failed",
                            "run.cancelled",
                        }:
                            snapshot = await self.get_run(work, external_run_id)
                            yield AgentProviderEvent(
                                kind=(
                                    AgentProviderEventKind.TERMINAL
                                    if snapshot.state.is_terminal
                                    else AgentProviderEventKind.STATE
                                ),
                                state=snapshot.state,
                                output=snapshot.output,
                                error_message=snapshot.error_message,
                                provider_data=snapshot.provider_data,
                            )
                            return
            except httpx.TimeoutException:
                normalized_error = AgentConnectorError(
                    "Hermes event stream timed out",
                    code="timeout",
                    retryable=True,
                )
            except httpx.RequestError:
                normalized_error = AgentConnectorError(
                    "Hermes event stream is unavailable",
                    code="endpoint_unavailable",
                    retryable=True,
                )
            if normalized_error is not None:
                raise normalized_error
        finally:
            self._watched_runs.discard(watcher_key)

    async def get_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> AgentRunSnapshot:
        target = _resolved_target(work)
        path = f"/v1/runs/{quote(external_run_id, safe='')}"
        payload = await self._request_json(
            "GET",
            target,
            path,
            not_found_is_unknown=True,
        )
        if payload.get("_not_found") is True:
            return AgentRunSnapshot(state=ConnectorRunState.UNKNOWN)
        raw_status = payload.get("status")
        if not isinstance(raw_status, str):
            raise _protocol_error("Hermes run snapshot has a malformed status")
        state = _STATE_MAP.get(raw_status, ConnectorRunState.UNKNOWN)
        safe_provider_status = payload.get(_SAFE_PROVIDER_STATUS_KEY)
        if not isinstance(safe_provider_status, str):
            safe_provider_status = None
        output = payload.get("output")
        error_message = payload.get("error")
        return AgentRunSnapshot(
            state=state,
            output=output if isinstance(output, str) else None,
            error_message=(
                _redact_text(error_message, credential=None)
                if isinstance(error_message, str)
                else None
            ),
            provider_data=_snapshot_provider_data(
                payload,
                provider_status=(
                    safe_provider_status
                    if state is ConnectorRunState.UNKNOWN or raw_status == "stopping"
                    else None
                ),
            ),
        )

    async def get_session_history(
        self,
        work: AgentRunWorkItem,
    ) -> Sequence[AgentHistoryMessage]:
        session_id = quote(work.task.external_session_id, safe="")
        payload = await self._request_json(
            "GET",
            _resolved_target(work),
            f"/api/sessions/{session_id}/messages?order=latest&limit=64",
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise _protocol_error("Hermes session history response is malformed")
        projected: list[AgentHistoryMessage] = []
        for item in data:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            if not content:
                continue
            projected.append(
                AgentHistoryMessage(
                    role=role,
                    content=content[:_MAX_HISTORY_MESSAGE_CHARS],
                )
            )
        return _bound_history(projected)

    async def respond_approval(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
        choice: str,
        resolve_all: bool,
    ) -> None:
        if choice not in _APPROVAL_CHOICES:
            raise AgentConnectorError(
                "Approval choice is not supported by Hermes",
                code="invalid_request",
            )
        await self._request_json(
            "POST",
            _resolved_target(work),
            f"/v1/runs/{quote(external_run_id, safe='')}/approval",
            json_body={"choice": choice, "resolve_all": resolve_all},
        )

    async def steer_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
        input: str,
    ) -> None:
        if work.run.status is not AgentRunStatus.RUNNING:
            raise AgentConnectorError(
                "Only a locally running Agent run can be steered",
                code="invalid_request",
            )
        if not isinstance(input, str) or not input.strip():
            raise AgentConnectorError(
                "Steer input must be non-empty",
                code="invalid_request",
            )
        await self._request_json(
            "POST",
            _resolved_target(work),
            f"/v1/runs/{quote(external_run_id, safe='')}/steer",
            json_body={"input": input},
        )

    async def stop_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> None:
        await self._request_json(
            "POST",
            _resolved_target(work),
            f"/v1/runs/{quote(external_run_id, safe='')}/stop",
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request_json(
        self,
        method: Literal["GET", "POST"],
        target: AgentResolvedTarget,
        path: str,
        *,
        json_body: object = _NO_BODY,
        dispatch: bool = False,
        session_scope: str | None = None,
        expected_status: int | None = None,
        not_found_is_unknown: bool = False,
    ) -> dict[str, Any]:
        headers, credential = self._headers(target)
        if session_scope is not None:
            headers["X-Hermes-Session-Key"] = session_scope
        content: bytes | None = None
        if json_body is not _NO_BODY:
            content = _encode_json_body(json_body)
            if len(content) > _MAX_JSON_BODY_BYTES:
                raise AgentConnectorError(
                    "Hermes request body exceeds the size limit",
                    code="invalid_request",
                )
            headers["Content-Type"] = "application/json"
        normalized_error: AgentConnectorError | None = None
        try:
            async with self._client.stream(
                method,
                self._url(target, path),
                headers=headers,
                timeout=self._request_timeout_seconds,
                follow_redirects=False,
                content=content,
            ) as response:
                if not_found_is_unknown and response.status_code == 404:
                    return {"_not_found": True}
                if not 200 <= response.status_code < 300:
                    body = await _read_bounded_error_body(response)
                    raise _http_error(
                        response,
                        body,
                        credential=credential,
                        dispatch=dispatch,
                    )
                body_error: AgentConnectorError | None = None
                try:
                    body = await _read_bounded_body(response)
                except AgentConnectorError as error:
                    if dispatch:
                        body_error = AgentConnectorError(
                            str(error),
                            code="dispatch_unknown",
                            dispatch_uncertain=True,
                        )
                    else:
                        body_error = error
                    body = b""
                if body_error is not None:
                    raise body_error
                if (
                    expected_status is not None
                    and response.status_code != expected_status
                ):
                    raise AgentConnectorError(
                        "Hermes returned an unexpected success status",
                        code="dispatch_unknown" if dispatch else "protocol_error",
                        dispatch_uncertain=dispatch,
                    )
                payload = _decode_json_payload(body, dispatch=dispatch)
                return _sanitize_provider_payload(payload, credential=credential)
        except httpx.TimeoutException:
            normalized_error = AgentConnectorError(
                "Hermes request timed out",
                code="timeout",
                retryable=not dispatch,
                dispatch_uncertain=dispatch,
            )
        except httpx.RequestError:
            normalized_error = AgentConnectorError(
                "Hermes endpoint is unavailable",
                code="endpoint_unavailable",
                retryable=not dispatch,
                dispatch_uncertain=dispatch,
            )
        if normalized_error is not None:
            raise normalized_error
        raise RuntimeError("Hermes request exited without a response")

    def _headers(self, target: AgentResolvedTarget) -> tuple[dict[str, str], str]:
        credential = self._credential(target)
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
        }, credential

    def _credential(self, target: AgentResolvedTarget) -> str:
        credential_ref = target.target.credential_ref
        if not credential_ref:
            raise AgentConnectorError(
                "Hermes target has no credential reference",
                code="authentication_required",
            )
        try:
            credential = self._credential_resolver.resolve(credential_ref)
        except AgentCredentialLookupError:
            credential = None
        if credential is None:
            raise AgentConnectorError(
                "Hermes credential is unavailable",
                code="authentication_required",
            )
        return credential

    @staticmethod
    def _url(target: AgentResolvedTarget, path: str) -> str:
        base = target.endpoint.base_url.rstrip("/")
        prefix = target.target.route_prefix.rstrip("/")
        return f"{base}{prefix}{path}"


def _resolved_target(work: AgentRunWorkItem) -> AgentResolvedTarget:
    return AgentResolvedTarget(target=work.target, endpoint=work.endpoint)


def _session_scope(work: AgentRunWorkItem) -> str:
    return f"fm-fish:{work.task.fish_id}:target:{work.target.id}"


def _normalize_base_url(base_url: str) -> str:
    if (
        not isinstance(base_url, str)
        or not base_url
        or any(character.isspace() for character in base_url)
    ):
        raise _invalid_target("Hermes base URL is malformed")
    parsed = None
    host = None
    parsed_safely = False
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        _ = parsed.port
        parsed_safely = True
    except ValueError:
        pass
    if not parsed_safely or parsed is None:
        raise _invalid_target("Hermes base URL is malformed")
    if parsed.scheme not in {"http", "https"} or not host or not parsed.netloc:
        raise _invalid_target("Hermes base URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise _invalid_target("Hermes base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise _invalid_target("Hermes base URL must not contain query or fragment")
    if parsed.scheme == "http" and not is_loopback_host(host):
        raise _invalid_target("Remote Hermes targets require HTTPS")
    return base_url.rstrip("/")


def _normalize_route_prefix(route_prefix: str) -> str:
    if not isinstance(route_prefix, str):
        raise _invalid_target("Hermes route prefix is malformed")
    normalized = route_prefix.rstrip("/")
    if normalized == "":
        return ""
    if not _ROUTE_PREFIX_RE.fullmatch(normalized):
        raise _invalid_target("Hermes route prefix is malformed")
    if any(segment in {".", ".."} for segment in normalized.split("/")[1:]):
        raise _invalid_target("Hermes route prefix is malformed")
    return normalized


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _invalid_target(message: str) -> AgentConnectorError:
    return AgentConnectorError(message, code="invalid_request")


def _encode_json_body(value: object) -> bytes:
    content: bytes | None = None
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        pass
    if content is None:
        raise AgentConnectorError(
            "Hermes request body is not JSON serializable",
            code="invalid_request",
        )
    return content


def _decode_json_payload(body: bytes, *, dispatch: bool) -> dict[str, Any]:
    payload: object | None = None
    decode_failed = False
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        decode_failed = True
    if decode_failed:
        raise AgentConnectorError(
            "Hermes returned malformed JSON",
            code="dispatch_unknown" if dispatch else "protocol_error",
            dispatch_uncertain=dispatch,
        )
    if not isinstance(payload, dict):
        raise AgentConnectorError(
            "Hermes returned a non-object JSON response",
            code="dispatch_unknown" if dispatch else "protocol_error",
            dispatch_uncertain=dispatch,
        )
    return payload


def _sanitize_provider_payload(
    payload: Mapping[str, object],
    *,
    credential: str,
) -> dict[str, Any]:
    sanitized = dict(payload)
    sanitized.pop(_OUTPUT_TRUNCATED_KEY, None)
    sanitized.pop(_SAFE_PROVIDER_STATUS_KEY, None)

    run_id = sanitized.get("run_id")
    if isinstance(run_id, str):
        safe_run_id = _redact_bounded_text(
            run_id,
            limit=_MAX_EXTERNAL_RUN_ID_CHARS,
            credential=credential,
        )
        if safe_run_id != run_id:
            sanitized["run_id"] = None

    output = sanitized.get("output")
    if isinstance(output, str):
        safe_output, output_truncated = _redact_and_bound_with_truncation(
            output,
            limit=_MAX_OUTPUT_CHARS,
            credential=credential,
        )
        sanitized["output"] = safe_output
        if output_truncated:
            sanitized[_OUTPUT_TRUNCATED_KEY] = True

    for field, limit in (
        ("error", _MAX_ERROR_TEXT_CHARS),
        ("pending_steer", _MAX_PENDING_STEER_CHARS),
    ):
        value = sanitized.get(field)
        if isinstance(value, str):
            sanitized[field] = _redact_bounded_text(
                value,
                limit=limit,
                credential=credential,
            )
    raw_status = sanitized.get("status")
    if isinstance(raw_status, str):
        sanitized[_SAFE_PROVIDER_STATUS_KEY] = _redact_bounded_text(
            raw_status,
            limit=_MAX_PROVIDER_STATUS_CHARS,
            credential=credential,
        )
    return sanitized


async def _read_bounded_body(response: httpx.Response) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_JSON_BODY_BYTES:
                raise _protocol_error("Hermes response body exceeds the size limit")
        except ValueError:
            pass
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_JSON_BODY_BYTES:
            raise _protocol_error("Hermes response body exceeds the size limit")
    return bytes(body)


async def _read_bounded_error_body(response: httpx.Response) -> bytes:
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            remaining = _MAX_JSON_BODY_BYTES - len(body)
            if remaining <= 0:
                break
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                break
    except Exception:
        pass
    return bytes(body)


def _http_error(
    response: httpx.Response,
    body: bytes,
    *,
    credential: str,
    dispatch: bool,
) -> AgentConnectorError:
    status = response.status_code
    if 300 <= status < 400:
        code = "protocol_error"
        retryable = False
        generic_message = "Hermes redirect responses are not allowed"
    elif status == 401:
        code = "authentication_required"
        retryable = False
        generic_message = "Hermes authentication failed"
    elif status == 403:
        code = "permission_denied"
        retryable = False
        generic_message = "Hermes denied the request"
    elif status == 429:
        code = "rate_limited"
        retryable = False
        generic_message = "Hermes rate limited the request"
    elif 400 <= status < 500:
        code = "invalid_request"
        retryable = False
        generic_message = f"Hermes rejected the request with HTTP {status}"
    else:
        code = "provider_error"
        retryable = True
        generic_message = f"Hermes request failed with HTTP {status}"
    message = _error_body_text(body, credential=credential)
    raw_retry_after = response.headers.get("Retry-After") if status == 429 else None
    retry_after = (
        _redact_bounded_text(
            raw_retry_after,
            limit=_MAX_RETRY_AFTER_CHARS,
            credential=credential,
        )
        if raw_retry_after is not None
        else None
    )
    return AgentConnectorError(
        message or generic_message,
        code=code,  # type: ignore[arg-type]
        retryable=retryable and not dispatch,
        dispatch_uncertain=dispatch and status >= 500,
        retry_after=retry_after,
    )


def _error_body_text(body: bytes, *, credential: str) -> str:
    raw = body.decode("utf-8", errors="replace")
    payload: object | None = None
    try:
        payload = json.loads(raw)
    except Exception:
        return ""
    try:
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                text = error["message"]
            elif isinstance(error, str):
                text = error
            else:
                text = str(payload.get("message") or "")
        else:
            text = ""
    except Exception:
        return ""
    return _redact_text(text, credential=credential)


def _redact_text(text: str, *, credential: str | None) -> str:
    return _redact_bounded_text(
        text,
        limit=_MAX_ERROR_TEXT_CHARS,
        credential=credential,
    )


def _redact_bounded_text(
    text: str,
    *,
    limit: int,
    credential: str | None,
) -> str:
    return _redact_and_bound_with_truncation(
        text,
        limit=limit,
        credential=credential,
    )[0]


def _redact_and_bound_with_truncation(
    text: str,
    *,
    limit: int,
    credential: str | None,
) -> tuple[str, bool]:
    redacted = text
    if credential:
        redacted = redacted.replace(credential, "[REDACTED]")
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", redacted)
    return redacted[:limit], len(redacted) > limit


def _protocol_error(message: str) -> AgentConnectorError:
    return AgentConnectorError(message, code="protocol_error")


def _compact_provider_data_size(value: Mapping[str, object]) -> int | None:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        return None
    return len(payload.encode("utf-8"))


def _bounded_provider_data(
    fields_by_priority: Sequence[tuple[str, object | None]],
) -> Mapping[str, object] | None:
    result: dict[str, object] = {}
    for key, value in fields_by_priority:
        if value is None or value == "":
            continue
        candidate = {**result, key: value}
        size = _compact_provider_data_size(candidate)
        if size is not None and size <= _MAX_PROVIDER_DATA_BYTES:
            result[key] = value
            continue
        if not isinstance(value, str):
            continue
        low = 1
        high = len(value)
        best: str | None = None
        while low <= high:
            middle = (low + high) // 2
            truncated = value[:middle]
            candidate = {**result, key: truncated}
            size = _compact_provider_data_size(candidate)
            if size is not None and size <= _MAX_PROVIDER_DATA_BYTES:
                best = truncated
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            result[key] = best
    return result or None


def _snapshot_provider_data(
    payload: Mapping[str, object],
    *,
    provider_status: str | None,
) -> Mapping[str, object] | None:
    pending_steer = payload.get("pending_steer")
    bounded_pending_steer = (
        pending_steer[:_MAX_PENDING_STEER_CHARS]
        if isinstance(pending_steer, str) and pending_steer
        else None
    )
    usage = payload.get("usage")
    bounded_usage: Mapping[str, object] | None = None
    if isinstance(usage, Mapping):
        bounded_usage = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance((value := usage.get(key)), (int, float))
        }
    return _bounded_provider_data(
        (
            ("provider_status", provider_status),
            (
                "output_truncated",
                True if payload.get(_OUTPUT_TRUNCATED_KEY) is True else None,
            ),
            ("pending_steer", bounded_pending_steer),
            ("usage", bounded_usage),
        )
    )


def _bound_history(
    messages: Sequence[AgentHistoryMessage],
) -> tuple[AgentHistoryMessage, ...]:
    selected_reversed: list[AgentHistoryMessage] = []
    remaining = _MAX_HISTORY_BYTES
    for message in reversed(messages):
        encoded = message.content.encode("utf-8")
        if len(encoded) <= remaining:
            selected_reversed.append(message)
            remaining -= len(encoded)
            continue
        if remaining > 0:
            truncated = encoded[:remaining].decode("utf-8", errors="ignore")
            if truncated:
                selected_reversed.append(replace(message, content=truncated))
        break
    return tuple(reversed(selected_reversed))


def _validate_outbound_history(
    messages: Sequence[AgentHistoryMessage],
) -> None:
    total_bytes = 0
    for message in messages:
        if message.role not in {"user", "assistant"}:
            raise AgentConnectorError(
                "Conversation history contains an unsupported role",
                code="invalid_request",
            )
        if not isinstance(message.content, str) or not message.content:
            raise AgentConnectorError(
                "Conversation history contains invalid content",
                code="invalid_request",
            )
        if len(message.content) > _MAX_HISTORY_MESSAGE_CHARS:
            raise AgentConnectorError(
                "Conversation history message exceeds the size limit",
                code="invalid_request",
            )
        total_bytes += len(message.content.encode("utf-8"))
        if total_bytes > _MAX_HISTORY_BYTES:
            raise AgentConnectorError(
                "Conversation history exceeds the size limit",
                code="invalid_request",
            )


async def _iter_sse_payloads(
    response: httpx.Response,
) -> AsyncIterator[tuple[dict[str, Any], str | None]]:
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        while True:
            boundary = _sse_boundary(buffer)
            if boundary is None:
                if len(buffer) > _MAX_SSE_FRAME_BYTES:
                    raise _protocol_error("Hermes SSE frame exceeds the size limit")
                break
            frame_end, delimiter_length = boundary
            frame = bytes(buffer[:frame_end])
            del buffer[: frame_end + delimiter_length]
            if len(frame) > _MAX_SSE_FRAME_BYTES:
                raise _protocol_error("Hermes SSE frame exceeds the size limit")
            parsed = _parse_sse_frame(frame)
            if parsed is not None:
                yield parsed
    if buffer:
        if len(buffer) > _MAX_SSE_FRAME_BYTES:
            raise _protocol_error("Hermes SSE frame exceeds the size limit")
        parsed = _parse_sse_frame(bytes(buffer))
        if parsed is not None:
            yield parsed


def _sse_boundary(buffer: bytearray) -> tuple[int, int] | None:
    lf = buffer.find(b"\n\n")
    crlf = buffer.find(b"\r\n\r\n")
    candidates = [
        (index, length) for index, length in ((lf, 2), (crlf, 4)) if index >= 0
    ]
    return min(candidates, default=None)


def _parse_sse_frame(frame: bytes) -> tuple[dict[str, Any], str | None] | None:
    text: str | None = None
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise _protocol_error("Hermes SSE frame is not valid UTF-8")
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        if not raw_line or raw_line.startswith(":"):
            continue
        field, separator, value = raw_line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if not data_lines:
        return None
    payload: object | None = None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        pass
    if payload is None:
        raise _protocol_error("Hermes SSE data is malformed JSON")
    if not isinstance(payload, dict):
        raise _protocol_error("Hermes SSE data must be a JSON object")
    return payload, event_name


def _approval_event(
    payload: Mapping[str, object],
    *,
    credential: str,
) -> AgentProviderEvent:
    raw_choices = payload.get("choices")
    choices = (
        tuple(
            choice
            for choice in raw_choices
            if isinstance(choice, str) and choice in _APPROVAL_CHOICES
        )
        if isinstance(raw_choices, list)
        else ()
    )
    provider_fields: list[tuple[str, object | None]] = []
    for key in ("command", "tool", "description", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            provider_fields.append((key, _redact_text(value, credential=credential)))
    smart_denied = payload.get("smart_denied")
    if isinstance(smart_denied, bool):
        provider_fields.append(("smart_denied", smart_denied))
    return AgentProviderEvent(
        kind=AgentProviderEventKind.APPROVAL,
        state=ConnectorRunState.WAITING_USER,
        approval_choices=choices,
        provider_data=_bounded_provider_data(provider_fields),
    )


__all__ = ["HermesConnector", "HermesConnectorFactory"]
