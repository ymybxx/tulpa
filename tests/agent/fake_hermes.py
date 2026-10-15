from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote

import httpx


class ChunkedSseStream(httpx.AsyncByteStream):
    """An in-memory chunked SSE body that still exercises HTTP streaming."""

    def __init__(self, content: bytes) -> None:
        midpoint = max(1, len(content) // 2)
        self._chunks = (content[:midpoint], content[midpoint:])

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self._chunks:
            if chunk:
                yield chunk

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _Scenario:
    kind: Literal["completed", "approval", "input_required"]
    message: str
    command: str | None = None
    sse_output: str | None = None


@dataclass(slots=True)
class _Run:
    run_id: str
    session_id: str
    scenario: _Scenario
    status: str
    output: str | None


class FakeHermes:
    """Stateful Hermes Runs API fake used only through httpx.MockTransport."""

    def __init__(self) -> None:
        self._scenarios: deque[_Scenario] = deque()
        self._runs: dict[str, _Run] = {}
        self.run_ids: list[str] = []
        self.run_requests: list[dict[str, object]] = []
        self.approval_requests: list[dict[str, object]] = []
        self.history_requests: list[str] = []
        self.event_requests: list[str] = []
        self.snapshot_requests: list[str] = []

    def enqueue_completed(
        self,
        message: str,
        *,
        sse_output: str | None = None,
    ) -> None:
        self._scenarios.append(_Scenario("completed", message, sse_output=sse_output))

    def enqueue_approval(self, *, command: str, completed_message: str) -> None:
        self._scenarios.append(
            _Scenario("approval", completed_message, command=command)
        )

    def enqueue_input_required(self, question: str) -> None:
        self._scenarios.append(_Scenario("input_required", question))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != "Bearer provider-key":
            return _json_response(401, {"error": {"code": "unauthorized"}})
        path = request.url.path
        if request.method == "GET" and path == "/v1/capabilities":
            return _json_response(
                200,
                {
                    "features": {
                        "run_submission": True,
                        "run_status": True,
                        "run_events_sse": True,
                        "session_resources": True,
                        "run_approval_response": True,
                        "run_steer": False,
                        "run_stop": False,
                    }
                },
            )
        if request.method == "POST" and path == "/v1/runs":
            return self._start_run(request)
        if request.method == "GET" and path.startswith("/api/sessions/"):
            return self._history(request)
        if path.startswith("/v1/runs/"):
            return self._run_request(request)
        return _json_response(404, {"error": {"code": "not_found"}})

    def _start_run(self, request: httpx.Request) -> httpx.Response:
        if not self._scenarios:
            return _json_response(500, {"error": {"code": "no_scenario"}})
        body = json.loads(request.content)
        if not isinstance(body, dict):
            return _json_response(400, {"error": {"code": "invalid_body"}})
        scenario = self._scenarios.popleft()
        run_id = f"fake-run-{len(self.run_ids) + 1}"
        output = _outcome(scenario)
        status = "waiting_for_approval" if scenario.kind == "approval" else "completed"
        session_id = body.get("session_id")
        if not isinstance(session_id, str):
            return _json_response(400, {"error": {"code": "invalid_session"}})
        self.run_ids.append(run_id)
        self.run_requests.append(dict(body))
        self._runs[run_id] = _Run(run_id, session_id, scenario, status, output)
        return _json_response(202, {"run_id": run_id, "status": "started"})

    def _history(self, request: httpx.Request) -> httpx.Response:
        prefix = "/api/sessions/"
        suffix = "/messages"
        path = request.url.path
        if not path.endswith(suffix):
            return _json_response(404, {"error": {"code": "not_found"}})
        session_id = unquote(path[len(prefix) : -len(suffix)])
        self.history_requests.append(session_id)
        return _json_response(
            200,
            {"object": "list", "session_id": session_id, "data": []},
        )

    def _run_request(self, request: httpx.Request) -> httpx.Response:
        tail = request.url.path.removeprefix("/v1/runs/")
        if tail.endswith("/events"):
            return self._events(request, unquote(tail.removesuffix("/events")))
        if tail.endswith("/approval"):
            return self._approval(request, unquote(tail.removesuffix("/approval")))
        run_id = unquote(tail)
        run = self._runs.get(run_id)
        if request.method != "GET" or run is None:
            return _json_response(404, {"error": {"code": "run_not_found"}})
        self.snapshot_requests.append(run_id)
        return _json_response(
            200,
            {
                "object": "hermes.run",
                "run_id": run_id,
                "status": run.status,
                "output": run.output,
            },
        )

    def _events(self, request: httpx.Request, run_id: str) -> httpx.Response:
        run = self._runs.get(run_id)
        if request.method != "GET" or run is None:
            return _json_response(404, {"error": {"code": "run_not_found"}})
        self.event_requests.append(run_id)
        if run.status == "completed":
            output = run.scenario.sse_output or run.output
            payload = {"event": "run.completed", "output": output}
            event = "run.completed"
        else:
            payload = {
                "event": "approval.request",
                "command": run.scenario.command,
                "choices": ["once", "deny"],
            }
            event = "approval.request"
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content = f"event: {event}\ndata: {data}\n\n".encode()
        return httpx.Response(
            200,
            stream=ChunkedSseStream(content),
            headers={"Content-Type": "text/event-stream"},
        )

    def _approval(self, request: httpx.Request, run_id: str) -> httpx.Response:
        run = self._runs.get(run_id)
        if request.method != "POST" or run is None:
            return _json_response(404, {"error": {"code": "run_not_found"}})
        body = json.loads(request.content)
        if not isinstance(body, dict):
            return _json_response(400, {"error": {"code": "invalid_body"}})
        self.approval_requests.append(
            {
                "run_id": run_id,
                "choice": body.get("choice"),
                "resolve_all": body.get("resolve_all"),
            }
        )
        run.status = "completed"
        run.output = _outcome(run.scenario)
        return _json_response(200, {"accepted": True})


def _outcome(scenario: _Scenario) -> str:
    disposition = "input_required" if scenario.kind == "input_required" else "completed"
    return json.dumps(
        {"disposition": disposition, "message": scenario.message},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )


__all__ = ["ChunkedSseStream", "FakeHermes"]
