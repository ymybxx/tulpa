from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fish_memory.home.models import (
    HomeAction,
    HomeEntity,
    HomeOperation,
    HomeQueryCategory,
)


@dataclass(frozen=True, slots=True)
class ServiceCall:
    service: str
    data: dict[str, object]


@dataclass(frozen=True, slots=True)
class StateConfirmation:
    expected_states: frozenset[str]


@dataclass(frozen=True, slots=True)
class CoverMovementConfirmation:
    initial_position: float
    target_position: int


Confirmation = StateConfirmation | CoverMovementConfirmation | None


class DomainAdapter:
    """Translate one Home Assistant entity domain into executable behavior.

    Adapters keep the domain contract in one place: advertised operations,
    service calls, target-state checks, and post-dispatch confirmation.
    """

    domain: str
    default_operations: frozenset[HomeOperation]
    active_states: frozenset[str] = frozenset()
    interesting_states: frozenset[str] = frozenset()
    abnormal_states: frozenset[str] = frozenset()

    def operations(
        self,
        attributes: Mapping[str, object],
        service_schemas: Mapping[str, frozenset[str]],
    ) -> frozenset[HomeOperation]:
        return self.default_operations

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        raise ValueError(
            f"{action.operation.value} is unsupported for domain: {self.domain}"
        )

    def target_is_satisfied(
        self,
        action: HomeAction,
        entity: HomeEntity,
    ) -> bool:
        return False

    def action_is_applicable(
        self,
        action: HomeAction,
        entity: HomeEntity,
    ) -> bool:
        return True

    def confirmation(
        self,
        action: HomeAction,
        entity: HomeEntity,
    ) -> Confirmation:
        return None

    def control_health(
        self,
        entity: HomeEntity,
        operation: HomeOperation,
    ) -> str | None:
        if entity.state == "unavailable":
            return "unavailable"
        if entity.state == "unknown" and not (
            entity.domain == "button" and operation is HomeOperation.ACTIVATE
        ):
            return "unknown"
        if entity.state in self.abnormal_states:
            return entity.state
        return None

    def query_category(self, entity: HomeEntity) -> HomeQueryCategory:
        if entity.state in {"unavailable", "unknown"} | self.abnormal_states:
            return HomeQueryCategory.ABNORMAL
        if entity.state in self.active_states:
            return HomeQueryCategory.ACTIVE
        if entity.state in self.interesting_states:
            return HomeQueryCategory.INTERESTING
        return HomeQueryCategory.INACTIVE


def percentage(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("percentage must be numeric")
    rounded = round(value)
    if not 0 <= rounded <= 100:
        raise ValueError("percentage must be between 0 and 100")
    return rounded


def numeric_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def numeric_action_value(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def state_matches(value: str, expected: frozenset[str] | None) -> bool:
    return expected is not None and value in expected


def toggle_target(current_state: str) -> frozenset[str] | None:
    if current_state == "on":
        return frozenset({"off"})
    if current_state == "off":
        return frozenset({"on"})
    return None
