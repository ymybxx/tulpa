from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from fish_memory.home.adapters import (
    CoverMovementConfirmation,
    StateConfirmation,
    adapter_for_domain,
)
from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.client import HomeActionUncertainError, HomeAssistantError
from fish_memory.home.models import (
    HomeAction,
    HomeEntity,
    HomeExecutionResult,
    HomeOperation,
)


class HomeActionClient(Protocol):
    @property
    def catalog(self) -> HomeEntityCatalog: ...

    async def call_service(
        self,
        domain: str,
        service: str,
        *,
        service_data: Mapping[str, object] | None = None,
        target: Mapping[str, object] | None = None,
    ) -> Any: ...

    async def wait_for_state(
        self,
        entity_id: str,
        expected_states: frozenset[str],
        *,
        timeout_seconds: float,
    ) -> str | None: ...

    async def wait_for_cover_movement(
        self,
        entity_id: str,
        initial_position: float,
        target_position: int,
        *,
        timeout_seconds: float,
    ) -> str | None: ...


class HomeActionExecutor:
    def __init__(
        self,
        client: HomeActionClient,
        *,
        confirmation_timeout: float = 3.0,
    ) -> None:
        self._client = client
        self._confirmation_timeout = confirmation_timeout

    def target_is_satisfied(self, action: HomeAction) -> bool:
        entity = self._validated_entity(action)
        return adapter_for_domain(entity.domain).target_is_satisfied(action, entity)

    def action_is_applicable(self, action: HomeAction) -> bool:
        entity = self._validated_entity(action)
        return adapter_for_domain(entity.domain).action_is_applicable(action, entity)

    async def execute(self, action: HomeAction) -> HomeExecutionResult:
        entity = self._validated_entity(action)
        if action.operation is HomeOperation.QUERY:
            return HomeExecutionResult(
                accepted=True,
                confirmed=True,
                state=entity.state,
            )
        if entity.state == "unavailable" or (
            entity.state == "unknown" and entity.domain != "button"
        ):
            return HomeExecutionResult(
                accepted=False,
                state=entity.state,
                error="设备当前不可用",
            )

        adapter = adapter_for_domain(entity.domain)
        if adapter.target_is_satisfied(action, entity):
            return HomeExecutionResult(
                accepted=True,
                confirmed=True,
                state=entity.state,
            )

        call = adapter.service_call(action, entity)
        try:
            response = await self._call_service(entity, call.service, call.data)
        except HomeActionUncertainError:
            raise
        except HomeAssistantError:
            if not _can_pause_after_stop_failure(action, entity):
                raise
            # Some integrations advertise STOP but reject it at runtime. PAUSE
            # preserves the user's intent to silence playback without relying on
            # integration-specific entity IDs or error strings.
            fallback_action = HomeAction(entity.entity_id, HomeOperation.PAUSE)
            call = adapter.service_call(fallback_action, entity)
            response = await self._call_service(entity, call.service, call.data)
        # Keep the original STOP confirmation semantics. The fallback exists
        # precisely because the provider's advertised capability contract is
        # unreliable, so waiting for its PAUSED state would add latency without
        # providing dependable confirmation.
        confirmation = adapter.confirmation(action, entity)
        if isinstance(confirmation, CoverMovementConfirmation):
            observed = await self._client.wait_for_cover_movement(
                entity.entity_id,
                confirmation.initial_position,
                confirmation.target_position,
                timeout_seconds=self._confirmation_timeout,
            )
            return _confirmed_result(
                observed=observed,
                response=response,
                domain=entity.domain,
                service=call.service,
            )
        if isinstance(confirmation, StateConfirmation):
            observed = await self._client.wait_for_state(
                entity.entity_id,
                confirmation.expected_states,
                timeout_seconds=self._confirmation_timeout,
            )
            return _confirmed_result(
                observed=observed,
                response=response,
                domain=entity.domain,
                service=call.service,
            )
        return HomeExecutionResult(
            accepted=True,
            response=response,
            service_domain=entity.domain,
            service_name=call.service,
        )

    async def _call_service(
        self,
        entity: HomeEntity,
        service: str,
        service_data: Mapping[str, object],
    ) -> Any:
        return await self._client.call_service(
            entity.domain,
            service,
            service_data=service_data,
            target={"entity_id": entity.entity_id},
        )

    def _validated_entity(self, action: HomeAction) -> HomeEntity:
        entity = self._client.catalog.get(action.entity_id)
        if entity is None:
            raise ValueError(f"unknown Home Assistant entity: {action.entity_id}")
        if action.operation not in entity.operations:
            raise ValueError(
                f"{action.entity_id} does not support {action.operation.value}"
            )
        return entity


def _can_pause_after_stop_failure(action: HomeAction, entity: HomeEntity) -> bool:
    return (
        entity.domain == "media_player"
        and action.operation is HomeOperation.STOP
        and HomeOperation.PAUSE in entity.operations
    )


def _confirmed_result(
    *,
    observed: str | None,
    response: object,
    domain: str,
    service: str,
) -> HomeExecutionResult:
    return HomeExecutionResult(
        accepted=True,
        confirmed=observed is not None,
        state=observed,
        response=response,
        service_domain=domain,
        service_name=service,
    )
