"""Which outside service broke a turn, and how, in terms the user can act on.

Each service client marks its own errors once, where it calls out
(`noted`); everything in between carries the error along unchanged. Whoever
ends a turn asks `failure_of` and passes the answer on as data; nothing on
the way decides what a particular failure means.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import httpx
import openai

Service = Literal["jev", "chat", "reasoning", "voice"]
Kind = Literal[
    "timeout", "connection", "key", "quota", "rate_limit", "rejected", "error"
]
SERVICES: tuple[Service, ...] = ("jev", "chat", "reasoning", "voice")
KINDS: tuple[Kind, ...] = (
    "timeout",
    "connection",
    "key",
    "quota",
    "rate_limit",
    "rejected",
    "error",
)
# Problems that stay until the user changes something.
LASTING: frozenset[Kind] = frozenset({"key", "quota"})

_SERVICE_ATTRIBUTE = "_fish_memory_service"

# Words providers use for an empty balance or a spent quota. Some send these
# with 429, some with 400 or 403.
_QUOTA_WORDS = (
    "quota",
    "arrearage",
    "overdue",
    "balance",
    "insufficient",
    "billing",
    "欠费",
    "余额",
    "额度",
)
_KEY_WORDS = ("apikey", "api_key", "api key", "invalidkey", "unauthorized")
# A provider closed to where the request comes from (e.g. Jev's 451, OpenAI's
# unsupported_country_region_territory): the network or proxy is the thing to
# fix, not what was said.
_REGION_WORDS = (
    "not available in your region",
    "unsupported_country",
    "region not supported",
)
# DashScope speech error codes (task-failed header).
_PROVIDER_CODES: dict[str, Kind] = {
    "invalidapikey": "key",
    "accessdenied": "key",
    "arrearage": "quota",
    "throttling": "rate_limit",
    "invalidparameter": "rejected",
    "datainspectionfailed": "rejected",
}


@dataclass(frozen=True, slots=True)
class ServiceFailure:
    service: Service
    kind: Kind

    @property
    def code(self) -> str:
        return f"{self.service}:{self.kind}"

    @classmethod
    def parse(cls, value: object) -> ServiceFailure | None:
        if not isinstance(value, str) or value.count(":") != 1:
            return None
        service, kind = value.split(":")
        if service not in SERVICES or kind not in KINDS:
            return None
        return cls(service, kind)  # type: ignore[arg-type]


class ProviderError(RuntimeError):
    """A provider's own refusal, with the code it gave, if any."""

    def __init__(self, message: str, code: object = None) -> None:
        super().__init__(message)
        self.code = code if isinstance(code, str) else None


class MissingKey(RuntimeError):
    """The service has no key configured."""


class ReportedFailure(RuntimeError):
    """A failure another process already classified, as its code."""

    def __init__(self, message: str, failure: object = None) -> None:
        super().__init__(message)
        self.failure = ServiceFailure.parse(failure)


def noted(error: BaseException, service: Service) -> BaseException:
    """Mark `error` as coming from `service` and remember lasting problems."""
    if isinstance(error, Exception) and not isinstance(error, asyncio.CancelledError):
        if getattr(error, _SERVICE_ATTRIBUTE, None) is None:
            try:
                setattr(error, _SERVICE_ATTRIBUTE, service)
            except AttributeError:  # pragma: no cover - slotted exceptions
                pass
        health.failed(ServiceFailure(service, kind_of(error)))
    return error


def succeeded(service: Service) -> None:
    health.succeeded(service)


def failure_of(error: BaseException | None) -> ServiceFailure | None:
    """The first marked service failure behind `error`, if any."""
    for item in _chain(error):
        if isinstance(item, ReportedFailure) and item.failure is not None:
            return item.failure
        service = getattr(item, _SERVICE_ATTRIBUTE, None)
        if service in SERVICES:
            return ServiceFailure(service, kind_of(item))
    return None


def classified(error: BaseException, service: Service) -> ServiceFailure:
    return failure_of(error) or ServiceFailure(service, kind_of(error))


def kind_of(error: BaseException) -> Kind:
    if isinstance(error, MissingKey):
        return "key"
    if isinstance(
        error,
        (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException),
    ) or isinstance(error, openai.APITimeoutError):
        return "timeout"
    if isinstance(error, openai.APIConnectionError) or isinstance(
        error, (httpx.TransportError, ConnectionError)
    ):
        return "connection"
    if isinstance(error, openai.APIStatusError):
        return _status_kind(error.status_code, _text(error.body, error.message))
    if isinstance(error, httpx.HTTPStatusError):
        return _status_kind(error.response.status_code, _response_text(error))
    code = getattr(error, "code", None)
    if isinstance(code, str):
        lowered = code.lower()
        for prefix, kind in _PROVIDER_CODES.items():
            if lowered.startswith(prefix):
                return kind
    text = str(error).lower()
    if any(word in text for word in _QUOTA_WORDS):
        return "quota"
    return "error"


def _status_kind(status: int, text: str) -> Kind:
    lowered = text.lower()
    if status == 451 or any(word in lowered for word in _REGION_WORDS):
        return "connection"
    if any(word in lowered for word in _QUOTA_WORDS) or status == 402:
        return "quota"
    if status in {401, 403} or any(word in lowered for word in _KEY_WORDS):
        return "key"
    if status == 429:
        return "rate_limit"
    if status in {408, 504}:
        return "timeout"
    if 400 <= status < 500:
        return "rejected"
    return "error"


def _text(body: object, message: str) -> str:
    try:
        return message + " " + json.dumps(body, ensure_ascii=False)
    except (TypeError, ValueError):
        return message


def _response_text(error: httpx.HTTPStatusError) -> str:
    try:
        return error.response.text[:2000]
    except Exception:  # pragma: no cover - unread streaming body
        return ""


def _chain(error: BaseException | None) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending = [error] if error is not None else []
    while pending:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        for linked in (item.__cause__, item.__context__):
            if linked is not None:
                pending.append(linked)


class ServiceHealth:
    """Lasting problems per service, cleared by that service's next success."""

    def __init__(self) -> None:
        self._problems: dict[Service, Kind] = {}
        self._listeners: list[Callable[[], None]] = []

    def problems(self) -> tuple[ServiceFailure, ...]:
        return tuple(ServiceFailure(s, k) for s, k in self._problems.items())

    def failed(self, failure: ServiceFailure) -> None:
        if failure.kind in LASTING and self._problems.get(failure.service) != (
            failure.kind
        ):
            self._problems[failure.service] = failure.kind
            self._changed()

    def succeeded(self, service: Service) -> None:
        if self._problems.pop(service, None) is not None:
            self._changed()

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def _changed(self) -> None:
        for listener in tuple(self._listeners):
            listener()


health = ServiceHealth()

# How TA names a failure when it reports back on work it could not finish.
SERVICE_NAMES: dict[Service, str] = {
    "jev": "陪伴大脑 Jev",
    "chat": "对话大脑",
    "reasoning": "推理大脑",
    "voice": "说话的声音",
}


def reason(failure: ServiceFailure) -> str:
    name = SERVICE_NAMES[failure.service]
    # "陪伴大脑 Jev 没有及时回应": a space after a Latin name, as in the UI.
    lead = name + " " if name[-1].isascii() else name
    return {
        "timeout": f"{lead}没有及时回应",
        "connection": f"连不上{name}",
        "key": f"{lead}的 Key 不对或已失效",
        "quota": f"{lead}的额度可能用完了",
        "rate_limit": f"{lead}说请求太频繁",
        "rejected": f"{lead}不接受这次的内容",
        "error": f"{lead}出错了",
    }[failure.kind]
