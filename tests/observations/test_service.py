from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from fish_memory.clock import MutableClock
from fish_memory.observations.models import (
    AppRecordingPolicy,
    ImportantObservationEvent,
    Observation,
    ObservationKind,
    ObservationQuery,
    StateObservation,
)
from fish_memory.observations.service import (
    ObservationRepository,
    ObservationService,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeRepository:
    def __init__(self) -> None:
        self.states: list[tuple[UUID, StateObservation]] = []
        self.events: list[tuple[UUID, ImportantObservationEvent]] = []
        self.queries: list[tuple[UUID, ObservationQuery]] = []
        self.latest_state_calls: list[tuple[UUID, str, str, str]] = []
        self.latest_state: Observation | None = None
        self.app_recording_calls: list[tuple[UUID, str]] = []
        self.app_recording_enabled = True

    async def record_state(
        self,
        fish_id: UUID,
        state: StateObservation,
    ) -> Observation | None:
        self.states.append((fish_id, state))
        return None

    async def record_event(
        self,
        fish_id: UUID,
        event: ImportantObservationEvent,
    ) -> Observation | None:
        self.events.append((fish_id, event))
        return None

    async def query_observations(
        self,
        fish_id: UUID,
        query: ObservationQuery,
    ) -> tuple[Observation, ...]:
        self.queries.append((fish_id, query))
        return ()

    async def get_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> Observation | None:
        return None

    async def get_latest_state_observation(
        self,
        fish_id: UUID,
        *,
        source: str,
        observation_type: str,
        scope_key: str,
    ) -> Observation | None:
        self.latest_state_calls.append((fish_id, source, observation_type, scope_key))
        return self.latest_state

    async def delete_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> bool:
        return False

    async def set_app_recording_policy(
        self,
        fish_id: UUID,
        app_bundle_id: str,
        *,
        recording_enabled: bool,
        app_name: str | None = None,
    ) -> AppRecordingPolicy:
        raise NotImplementedError

    async def list_app_recording_policies(
        self,
        fish_id: UUID,
    ) -> tuple[AppRecordingPolicy, ...]:
        return ()

    async def is_app_recording_enabled(
        self,
        fish_id: UUID,
        app_bundle_id: str,
    ) -> bool:
        self.app_recording_calls.append((fish_id, app_bundle_id))
        return self.app_recording_enabled


@pytest.mark.asyncio
async def test_service_stamps_structured_observations_without_conversation_calls() -> (
    None
):
    now = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    repository = FakeRepository()
    service = ObservationService(
        cast(ObservationRepository, repository),
        clock=MutableClock(now),
    )

    await service.observe_state(
        FISH_ID,
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="editing:fish-memory",
        summary="Editing Fish Memory",
        app_bundle_id="com.microsoft.VSCode",
    )
    await service.record_important_event(
        FISH_ID,
        source="accessibility",
        observation_type="error",
        semantic_key="test-failed",
        summary="A focused test failed",
    )

    assert repository.states[0][1].observed_at == now
    assert repository.states[0][1].app_bundle_id == "com.microsoft.vscode"
    assert repository.events[0][1].occurred_at == now


@pytest.mark.asyncio
async def test_service_builds_isolated_time_app_and_type_query() -> None:
    repository = FakeRepository()
    service = ObservationService(cast(ObservationRepository, repository))
    start = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    end = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)

    await service.query(
        FISH_ID,
        start_at=start,
        end_at=end,
        app_bundle_id="COM.APP.TEST",
        observation_type="error",
        limit=20,
    )

    query = repository.queries[0][1]
    assert query.start_at == start
    assert query.end_at == end
    assert query.app_bundle_id == "com.app.test"
    assert query.observation_type == "error"
    assert query.limit == 20


@pytest.mark.asyncio
async def test_service_resolves_latest_value_in_exact_state_stream() -> None:
    repository = FakeRepository()
    service = ObservationService(cast(ObservationRepository, repository))
    observed_at = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    current = Observation(
        id=UUID(int=10),
        sequence=10,
        fish_id=FISH_ID,
        kind=ObservationKind.STATE,
        source="workspace",
        observation_type="activity",
        scope_key="active-app",
        semantic_key="com.example.editor",
        summary="正在使用 Editor",
        payload={},
        app_bundle_id="com.example.editor",
        app_name="Editor",
        started_at=observed_at,
        ended_at=observed_at,
        created_at=observed_at,
        updated_at=observed_at,
    )
    repository.latest_state = current

    assert await service.latest_for_stream(FISH_ID, current) is current
    assert repository.latest_state_calls == [
        (FISH_ID, "workspace", "activity", "active-app")
    ]

    with pytest.raises(ValueError, match="does not belong"):
        await service.latest_for_stream(UUID(int=999), current)


@pytest.mark.asyncio
async def test_app_recording_preflight_validates_and_normalizes_ids() -> None:
    repository = FakeRepository()
    service = ObservationService(cast(ObservationRepository, repository))

    assert not await service.preflight_app_recording(FISH_ID, None)
    assert not await service.preflight_app_recording(FISH_ID, "")
    assert not await service.preflight_app_recording(FISH_ID, "not a bundle id")
    assert repository.app_recording_calls == []

    assert await service.preflight_app_recording(
        FISH_ID,
        " COM.Example.Private ",
    )
    repository.app_recording_enabled = False
    assert not await service.preflight_app_recording(
        FISH_ID,
        "com.example.private",
    )
    assert repository.app_recording_calls == [
        (FISH_ID, "com.example.private"),
        (FISH_ID, "com.example.private"),
    ]
