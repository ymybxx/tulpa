from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol
from uuid import UUID

from fish_memory.clock import Clock, SystemClock
from fish_memory.observations.models import (
    AppRecordingPolicy,
    ImportantObservationEvent,
    Observation,
    ObservationKind,
    ObservationQuery,
    StateObservation,
    normalize_app_bundle_id,
)


class ObservationRepository(Protocol):
    async def record_state(
        self,
        fish_id: UUID,
        state: StateObservation,
    ) -> Observation | None: ...

    async def record_event(
        self,
        fish_id: UUID,
        event: ImportantObservationEvent,
    ) -> Observation | None: ...

    async def get_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> Observation | None: ...

    async def get_latest_state_observation(
        self,
        fish_id: UUID,
        *,
        source: str,
        observation_type: str,
        scope_key: str,
    ) -> Observation | None: ...

    async def query_observations(
        self,
        fish_id: UUID,
        query: ObservationQuery,
    ) -> tuple[Observation, ...]: ...

    async def delete_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> bool: ...

    async def set_app_recording_policy(
        self,
        fish_id: UUID,
        app_bundle_id: str,
        *,
        recording_enabled: bool,
        app_name: str | None = None,
    ) -> AppRecordingPolicy: ...

    async def list_app_recording_policies(
        self,
        fish_id: UUID,
    ) -> tuple[AppRecordingPolicy, ...]: ...

    async def is_app_recording_enabled(
        self,
        fish_id: UUID,
        app_bundle_id: str,
    ) -> bool: ...


class ObservationService:
    """Small application boundary for durable, structured observations."""

    def __init__(
        self,
        repository: ObservationRepository,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or SystemClock()

    async def observe_state(
        self,
        fish_id: UUID,
        *,
        source: str,
        observation_type: str,
        scope_key: str,
        semantic_key: str,
        summary: str,
        payload: Mapping[str, object] | None = None,
        app_bundle_id: str | None = None,
        app_name: str | None = None,
        observed_at: datetime | None = None,
    ) -> Observation | None:
        return await self._repository.record_state(
            fish_id,
            StateObservation(
                source=source,
                observation_type=observation_type,
                scope_key=scope_key,
                semantic_key=semantic_key,
                summary=summary,
                payload={} if payload is None else payload,
                app_bundle_id=app_bundle_id,
                app_name=app_name,
                observed_at=observed_at or self._clock.now(),
            ),
        )

    async def record_important_event(
        self,
        fish_id: UUID,
        *,
        source: str,
        observation_type: str,
        semantic_key: str,
        summary: str,
        payload: Mapping[str, object] | None = None,
        app_bundle_id: str | None = None,
        app_name: str | None = None,
        occurred_at: datetime | None = None,
    ) -> Observation | None:
        return await self._repository.record_event(
            fish_id,
            ImportantObservationEvent(
                source=source,
                observation_type=observation_type,
                semantic_key=semantic_key,
                summary=summary,
                payload={} if payload is None else payload,
                app_bundle_id=app_bundle_id,
                app_name=app_name,
                occurred_at=occurred_at or self._clock.now(),
            ),
        )

    async def query(
        self,
        fish_id: UUID,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        app_bundle_id: str | None = None,
        observation_type: str | None = None,
        kind: ObservationKind | None = None,
        limit: int = 100,
    ) -> tuple[Observation, ...]:
        return await self._repository.query_observations(
            fish_id,
            ObservationQuery(
                start_at=start_at,
                end_at=end_at,
                app_bundle_id=app_bundle_id,
                observation_type=observation_type,
                kind=kind,
                limit=limit,
            ),
        )

    async def get(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> Observation | None:
        return await self._repository.get_observation(fish_id, observation_id)

    async def latest_for_stream(
        self,
        fish_id: UUID,
        observation: Observation,
    ) -> Observation | None:
        if observation.fish_id != fish_id:
            raise ValueError("observation does not belong to fish_id")
        if (
            observation.kind is not ObservationKind.STATE
            or observation.scope_key is None
        ):
            return None
        return await self._repository.get_latest_state_observation(
            fish_id,
            source=observation.source,
            observation_type=observation.observation_type,
            scope_key=observation.scope_key,
        )

    async def delete(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> bool:
        return await self._repository.delete_observation(fish_id, observation_id)

    async def set_app_recording(
        self,
        fish_id: UUID,
        app_bundle_id: str,
        *,
        enabled: bool,
        app_name: str | None = None,
    ) -> AppRecordingPolicy:
        return await self._repository.set_app_recording_policy(
            fish_id,
            app_bundle_id,
            recording_enabled=enabled,
            app_name=app_name,
        )

    async def list_app_recording_policies(
        self,
        fish_id: UUID,
    ) -> tuple[AppRecordingPolicy, ...]:
        return await self._repository.list_app_recording_policies(fish_id)

    async def preflight_app_recording(
        self,
        fish_id: UUID,
        app_bundle_id: str | None,
    ) -> bool:
        """Return an explicit allow decision for a valid foreground app.

        Missing or malformed identifiers are denied before reaching storage so
        future visual callers can safely require an affirmative response.
        """

        try:
            normalized_bundle_id = normalize_app_bundle_id(app_bundle_id)
        except ValueError:
            return False
        if normalized_bundle_id is None:
            return False
        return await self._repository.is_app_recording_enabled(
            fish_id,
            normalized_bundle_id,
        )
