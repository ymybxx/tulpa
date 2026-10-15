from datetime import UTC, datetime

import pytest

from fish_memory.observations.models import (
    ImportantObservationEvent,
    ObservationKind,
    ObservationQuery,
    StateObservation,
)


def test_state_requires_an_aware_timestamp_and_normalizes_bundle_id() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        StateObservation(
            source="workspace",
            observation_type="activity",
            scope_key="active-window",
            semantic_key="editing",
            summary="Editing code",
            observed_at=datetime(2026, 9, 1, 8, 0),
        )

    state = StateObservation(
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="editing",
        summary="Editing code",
        observed_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
        app_bundle_id=" COM.MICROSOFT.VSCODE ",
    )

    assert state.app_bundle_id == "com.microsoft.vscode"


@pytest.mark.parametrize(
    "payload",
    [
        {"screenshot": "small-but-still-raw"},
        {"nested": {"frame": [1, 2, 3]}},
        {"imageData": "small-but-still-raw"},
        {"rawOcr": ["unfiltered", "tokens"]},
        {"preview": "data:image/png;base64,abc"},
    ],
)
def test_observations_reject_raw_visual_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="raw screen or camera data"):
        ImportantObservationEvent(
            source="screen",
            observation_type="error",
            semantic_key="build-error",
            summary="Build failed",
            occurred_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
            payload=payload,
        )


def test_query_validates_time_window_and_limit() -> None:
    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    end = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="end_at"):
        ObservationQuery(start_at=start, end_at=end)
    with pytest.raises(ValueError, match="limit"):
        ObservationQuery(limit=501)

    query = ObservationQuery(
        observation_type="activity",
        kind=ObservationKind.STATE,
    )
    assert query.kind is ObservationKind.STATE


def test_observation_summaries_redact_credentials_at_the_domain_boundary() -> None:
    secret = "sk-1234567890abcdefghijklmnop"
    state = StateObservation(
        source="screen",
        observation_type="screen_context",
        scope_key="frontmost-window",
        semantic_key="editor",
        summary=f"Editing .env OPENAI_API_KEY={secret}",
        payload={"focusedValue": f"Authorization: Bearer {secret}"},
        observed_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
    )
    event = ImportantObservationEvent(
        source="screen",
        observation_type="screen_context",
        semantic_key="editor-event",
        summary="访问令牌：secret-value-123456",
        occurred_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
    )

    assert state.summary == "Editing .env OPENAI_API_KEY=[REDACTED]"
    assert secret not in state.summary
    assert state.payload == {"focusedValue": "Authorization=[REDACTED]"}
    assert event.summary == "访问令牌=[REDACTED]"
