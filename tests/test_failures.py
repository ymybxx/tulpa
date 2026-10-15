import asyncio

import httpx
import openai
import pytest

from fish_memory import failures
from fish_memory.failures import (
    MissingKey,
    ProviderError,
    ReportedFailure,
    ServiceFailure,
    ServiceHealth,
)

REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def status_error(status: int, body: object = None) -> openai.APIStatusError:
    response = httpx.Response(status, request=REQUEST, json=body or {})
    return openai.APIStatusError("provider said no", response=response, body=body)


@pytest.mark.parametrize(
    "error,kind",
    [
        (openai.APITimeoutError(request=REQUEST), "timeout"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (TimeoutError(), "timeout"),
        (openai.APIConnectionError(request=REQUEST), "connection"),
        (httpx.ConnectError("refused"), "connection"),
        (status_error(401), "key"),
        (status_error(403), "key"),
        (status_error(402), "quota"),
        (status_error(429, {"error": {"code": "insufficient_quota"}}), "quota"),
        (status_error(400, {"error": {"code": "Arrearage"}}), "quota"),
        (status_error(429), "rate_limit"),
        (status_error(400), "rejected"),
        (
            status_error(451, {"title": "Typesafe is not available in your region."}),
            "connection",
        ),
        (
            status_error(
                403, {"error": {"code": "unsupported_country_region_territory"}}
            ),
            "connection",
        ),
        (status_error(500), "error"),
        (MissingKey("no key"), "key"),
        (
            ProviderError("Please ensure input text is valid.", "InvalidParameter"),
            "rejected",
        ),
        (ProviderError("no", "Throttling.RateQuota"), "rate_limit"),
        (ProviderError("no", "InvalidApiKey"), "key"),
        (ValueError("Invalid Jev response"), "error"),
    ],
)
def test_kinds(error, kind):
    assert failures.kind_of(error) == kind


def test_http_status_errors_use_the_response():
    response = httpx.Response(401, request=REQUEST, text="unauthorized")
    error = httpx.HTTPStatusError("no", request=REQUEST, response=response)
    assert failures.kind_of(error) == "key"


def test_the_marked_service_is_found_behind_wrapping_errors():
    cause = failures.noted(httpx.ReadTimeout("slow"), "jev")
    try:
        try:
            raise cause
        except httpx.ReadTimeout as error:
            raise RuntimeError("answer generation failed") from error
    except RuntimeError as wrapped:
        assert failures.failure_of(wrapped) == ServiceFailure("jev", "timeout")
    group = ExceptionGroup("several", [ValueError("x"), cause])
    assert failures.failure_of(group) == ServiceFailure("jev", "timeout")
    assert failures.failure_of(ValueError("unmarked")) is None


def test_the_first_service_to_mark_an_error_keeps_it():
    error = httpx.ConnectError("refused")
    failures.noted(error, "reasoning")
    failures.noted(error, "chat")
    assert failures.failure_of(error) == ServiceFailure("reasoning", "connection")


def test_cancellation_is_not_a_failure():
    error = asyncio.CancelledError()
    failures.noted(error, "chat")
    assert failures.failure_of(error) is None


def test_a_failure_classified_elsewhere_travels_as_its_code():
    assert failures.failure_of(ReportedFailure("tts", "voice:rejected")) == (
        ServiceFailure("voice", "rejected")
    )
    assert failures.failure_of(ReportedFailure("tts", "nonsense")) is None
    assert ServiceFailure.parse("chat:quota") == ServiceFailure("chat", "quota")
    assert ServiceFailure.parse("chat:bored") is None
    assert ServiceFailure.parse(None) is None


def test_health_keeps_lasting_problems_until_that_service_works():
    health = ServiceHealth()
    changes: list[tuple[ServiceFailure, ...]] = []
    stop = health.subscribe(lambda: changes.append(health.problems()))
    health.failed(ServiceFailure("chat", "timeout"))
    assert health.problems() == () and changes == []
    health.failed(ServiceFailure("chat", "quota"))
    health.failed(ServiceFailure("chat", "quota"))
    health.failed(ServiceFailure("jev", "key"))
    assert changes == [
        (ServiceFailure("chat", "quota"),),
        (ServiceFailure("chat", "quota"), ServiceFailure("jev", "key")),
    ]
    health.succeeded("reasoning")
    health.succeeded("chat")
    assert health.problems() == (ServiceFailure("jev", "key"),)
    stop()
    health.succeeded("jev")
    assert len(changes) == 3


def test_reason_names_the_service_as_settings_do():
    assert (
        failures.reason(ServiceFailure("reasoning", "connection")) == "连不上推理大脑"
    )
    assert failures.reason(ServiceFailure("jev", "timeout")) == (
        "陪伴大脑 Jev 没有及时回应"
    )
    assert (
        failures.reason(ServiceFailure("chat", "quota")) == "对话大脑的额度可能用完了"
    )
