import pytest

from fish_memory.agent.outcome import (
    AgentOutcomeDisposition,
    normalize_agent_run_outcome,
    parse_agent_run_outcome,
)


@pytest.mark.parametrize(
    ("payload", "disposition", "message"),
    [
        (
            '{"disposition":"completed","message":"task finished"}',
            AgentOutcomeDisposition.COMPLETED,
            "task finished",
        ),
        (
            '{"disposition":"input_required","message":"which branch?"}',
            AgentOutcomeDisposition.INPUT_REQUIRED,
            "which branch?",
        ),
    ],
)
def test_parser_accepts_only_the_two_documented_outcomes(
    payload: str,
    disposition: AgentOutcomeDisposition,
    message: str,
) -> None:
    outcome = parse_agent_run_outcome(payload)

    assert outcome.disposition is disposition
    assert outcome.message == message


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        '{"disposition":"other","message":"x"}',
        '{"disposition":"completed","message":""}',
        '{"disposition":"completed","message":"   "}',
        '{"disposition":"completed","message":"x","extra":true}',
        ('{"disposition":"completed","message":"first","message":"second"}'),
        '{"disposition":"completed","message":1}',
        '{"disposition":"completed"}',
    ],
)
def test_parser_rejects_malformed_or_noncanonical_outcomes(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_agent_run_outcome(payload)


def test_parser_rejects_message_over_materialized_limit() -> None:
    payload = '{"disposition":"completed","message":"' + "x" * 32_001 + '"}'

    with pytest.raises(ValueError, match="message"):
        parse_agent_run_outcome(payload)


def test_malformed_terminal_output_is_delivered_once_as_safe_completed_text() -> None:
    raw = "Bearer provider-secret\ntoken=provider-secret\n" + "界" * 40_000

    outcome = normalize_agent_run_outcome(raw)

    assert outcome.disposition is AgentOutcomeDisposition.COMPLETED
    assert "provider-secret" not in outcome.message
    assert "[REDACTED]" in outcome.message
    assert len(outcome.message) <= 32_000


@pytest.mark.parametrize(
    "raw",
    [
        '{"password":"hunter2"}',
        '{"api_key":"sk-live-secret"}',
        "password='hunter2'",
        'api-key = "sk-live-secret"',
        "token: 'private-token'",
    ],
)
def test_malformed_terminal_output_redacts_quoted_secret_assignments(
    raw: str,
) -> None:
    outcome = normalize_agent_run_outcome(raw + "界" * 40_000)

    assert outcome.disposition is AgentOutcomeDisposition.COMPLETED
    assert "hunter2" not in outcome.message
    assert "sk-live-secret" not in outcome.message
    assert "private-token" not in outcome.message
    assert "[REDACTED]" in outcome.message
    assert len(outcome.message) <= 32_000


@pytest.mark.parametrize(
    ("raw", "secret_fragments"),
    [
        (r'{"password":"hunter\"2secret"}', ("hunter", "2secret")),
        (r'{"api_key":"folder\\sk-live-secret"}', ("folder", "sk-live-secret")),
        (r"password='hunter\'2secret'", ("hunter", "2secret")),
        (r'api_key="sk\"live-secret"', ("sk", "live-secret")),
    ],
)
def test_malformed_terminal_output_redacts_escaped_quoted_secret_without_suffix(
    raw: str,
    secret_fragments: tuple[str, ...],
) -> None:
    outcome = normalize_agent_run_outcome(raw + "界" * 40_000)

    assert outcome.disposition is AgentOutcomeDisposition.COMPLETED
    assert all(fragment not in outcome.message for fragment in secret_fragments)
    assert "[REDACTED]" in outcome.message
    assert len(outcome.message) <= 32_000


def test_empty_terminal_output_has_a_bounded_completed_fallback() -> None:
    outcome = normalize_agent_run_outcome("")

    assert outcome.disposition is AgentOutcomeDisposition.COMPLETED
    assert outcome.message
    assert len(outcome.message) <= 32_000
