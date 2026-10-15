from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class HomeIntentType(StrEnum):
    QUERY = "query"
    CONTROL = "control"


class HomeScopeKind(StrEnum):
    HOME = "home"
    AREA = "area"
    SPECIFIC = "specific"


class HomeSelection(StrEnum):
    ALL = "all"
    SPECIFIC = "specific"
    COUNT = "count"


class HomeStateFilter(StrEnum):
    INTERESTING = "interesting"
    ALL = "all"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ABNORMAL = "abnormal"


class HomeQueryCategory(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    INTERESTING = "interesting"
    ABNORMAL = "abnormal"


class HomeQueryDetail(StrEnum):
    SUMMARY = "summary"
    FULL = "full"


class HomeOperation(StrEnum):
    QUERY = "query"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    TOGGLE = "toggle"
    SET_BRIGHTNESS = "set_brightness"
    SET_COLOR_TEMPERATURE = "set_color_temperature"
    SET_TEMPERATURE = "set_temperature"
    PLAY = "play"
    PAUSE = "pause"
    OPEN = "open"
    CLOSE = "close"
    STOP = "stop"
    SET_POSITION = "set_position"
    SET_PERCENTAGE = "set_percentage"
    LOCK = "lock"
    UNLOCK = "unlock"
    START = "start"
    RETURN_TO_BASE = "return_to_base"
    ACTIVATE = "activate"


@dataclass(frozen=True, slots=True)
class HomeAreaFacet:
    area_id: str
    name: str


@dataclass(frozen=True, slots=True)
class HomeScopePlan:
    kind: HomeScopeKind
    area_indexes: tuple[int, ...]
    domains: tuple[str, ...]
    selection: HomeSelection
    count: int | None


@dataclass(frozen=True, slots=True)
class HomeControlPlan:
    scope: HomeScopePlan
    operation: HomeOperation
    value: str | int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class HomeQueryPlan:
    scope: HomeScopePlan
    state_filter: HomeStateFilter
    detail: HomeQueryDetail


class HomeActionStatus(StrEnum):
    ROUTING = "routing"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class HomeEntity:
    entity_id: str
    domain: str
    state: str
    attributes: Mapping[str, object]
    aliases: tuple[str, ...]
    area_id: str | None = None
    area_name: str | None = None
    device_id: str | None = None
    device_name: str | None = None
    operations: frozenset[HomeOperation] = frozenset()
    service_schemas: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class HomeCandidate:
    entity: HomeEntity
    score: int
    matched_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HomeQueryState:
    entity: HomeEntity
    category: HomeQueryCategory


@dataclass(frozen=True, slots=True)
class ExcludedHomeTarget:
    entity: HomeEntity
    reason: str


@dataclass(frozen=True, slots=True)
class ResolvedQueryScope:
    entities: tuple[HomeQueryState, ...]


@dataclass(frozen=True, slots=True)
class ResolvedControlScope:
    eligible: tuple[HomeCandidate, ...]
    excluded: tuple[ExcludedHomeTarget, ...]
    satisfied: tuple[HomeAction, ...]


@dataclass(frozen=True, slots=True)
class HomeRunOutcome:
    intent: HomeIntentType
    status: HomeActionStatus
    query: ResolvedQueryScope | None = None
    actions: tuple[HomeAction, ...] = ()
    excluded: tuple[ExcludedHomeTarget, ...] = ()
    satisfied: tuple[HomeAction, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HomeAction:
    entity_id: str
    operation: HomeOperation
    value: str | int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class HomeSetTarget:
    selection: str
    area: str | None
    domain: str | None
    count: int | None


@dataclass(frozen=True, slots=True)
class HomeSetCommand:
    target: HomeSetTarget
    operation: HomeOperation
    value: str | int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class HomeRouteDecision:
    commands: tuple[HomeAction | HomeSetCommand, ...]
    source: str
    skipped_commands: tuple[HomeAction, ...] = ()


@dataclass(frozen=True, slots=True)
class HomeExecutionResult:
    accepted: bool
    confirmed: bool = False
    state: str | None = None
    response: object | None = None
    error: str | None = None
    service_domain: str | None = None
    service_name: str | None = None


@dataclass(frozen=True, slots=True)
class HomeCatalogSnapshot:
    entities: Mapping[str, HomeEntity]
    ready: bool = True
