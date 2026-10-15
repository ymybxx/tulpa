from __future__ import annotations

from collections.abc import Mapping, Sequence

from fish_memory.home.adapters.base import (
    Confirmation,
    CoverMovementConfirmation,
    DomainAdapter,
    ServiceCall,
    StateConfirmation,
    numeric_action_value,
    numeric_value,
    percentage,
    state_matches,
    toggle_target,
)
from fish_memory.home.models import (
    HomeAction,
    HomeEntity,
    HomeOperation,
    HomeQueryCategory,
)


class LightAdapter(DomainAdapter):
    domain = "light"
    active_states = frozenset({"on"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.TURN_ON,
            HomeOperation.TURN_OFF,
            HomeOperation.TOGGLE,
            HomeOperation.SET_BRIGHTNESS,
            HomeOperation.SET_COLOR_TEMPERATURE,
        }
    )

    def operations(
        self,
        attributes: Mapping[str, object],
        service_schemas: Mapping[str, frozenset[str]],
    ) -> frozenset[HomeOperation]:
        operations = set(self.default_operations)
        if "turn_on" not in service_schemas:
            operations.discard(HomeOperation.SET_BRIGHTNESS)
        if not _supports_color_temperature(
            attributes,
            service_schemas.get("turn_on", frozenset()),
        ):
            operations.discard(HomeOperation.SET_COLOR_TEMPERATURE)
        return frozenset(operations)

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.SET_BRIGHTNESS:
            return ServiceCall("turn_on", {"brightness_pct": percentage(action.value)})
        if action.operation is HomeOperation.SET_COLOR_TEMPERATURE:
            kelvin = _color_temperature_kelvin(action.value, entity)
            fields = entity.service_schemas.get("turn_on", frozenset())
            if "color_temp_kelvin" in fields:
                return ServiceCall("turn_on", {"color_temp_kelvin": kelvin})
            if "color_temp" in fields:
                return ServiceCall("turn_on", {"color_temp": round(1_000_000 / kelvin)})
            raise ValueError("light does not expose a color temperature service field")
        return _power_service_call(action, self.domain)

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        if action.operation is HomeOperation.SET_BRIGHTNESS:
            brightness = numeric_value(entity.attributes.get("brightness"))
            return brightness is not None and round(
                brightness / 255 * 100
            ) == percentage(action.value)
        if action.operation is HomeOperation.SET_COLOR_TEMPERATURE:
            current = numeric_value(entity.attributes.get("color_temp_kelvin"))
            return current is not None and current == float(
                _color_temperature_kelvin(action.value, entity)
            )
        return state_matches(entity.state, _power_target(action, entity.state))

    def action_is_applicable(
        self,
        action: HomeAction,
        entity: HomeEntity,
    ) -> bool:
        if action.operation in {
            HomeOperation.SET_BRIGHTNESS,
            HomeOperation.SET_COLOR_TEMPERATURE,
        }:
            return entity.state == "on"
        return True

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        expected = _power_target(action, entity.state)
        if action.operation in {
            HomeOperation.SET_BRIGHTNESS,
            HomeOperation.SET_COLOR_TEMPERATURE,
        }:
            expected = frozenset({"on"})
        return _state_confirmation(expected)


class SwitchAdapter(DomainAdapter):
    domain = "switch"
    active_states = frozenset({"on"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.TURN_ON,
            HomeOperation.TURN_OFF,
            HomeOperation.TOGGLE,
        }
    )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        return _power_service_call(action, self.domain)

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        return state_matches(entity.state, _power_target(action, entity.state))

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        return _state_confirmation(_power_target(action, entity.state))


class ClimateAdapter(DomainAdapter):
    domain = "climate"
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.TURN_ON,
            HomeOperation.TURN_OFF,
            HomeOperation.SET_TEMPERATURE,
        }
    )

    def query_category(self, entity: HomeEntity) -> HomeQueryCategory:
        category = super().query_category(entity)
        if category is HomeQueryCategory.INACTIVE and entity.state != "off":
            return HomeQueryCategory.ACTIVE
        return category

    def operations(
        self,
        attributes: Mapping[str, object],
        service_schemas: Mapping[str, frozenset[str]],
    ) -> frozenset[HomeOperation]:
        operations = set(self.default_operations)
        if "turn_on" not in service_schemas:
            operations.discard(HomeOperation.TURN_ON)
        if "turn_off" not in service_schemas:
            operations.discard(HomeOperation.TURN_OFF)
        return frozenset(operations)

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.SET_TEMPERATURE:
            value = numeric_action_value(action.value, name="temperature")
            return ServiceCall("set_temperature", {"temperature": value})
        return _power_service_call(action, self.domain)

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        if action.operation is HomeOperation.SET_TEMPERATURE:
            current = numeric_value(entity.attributes.get("temperature"))
            target = numeric_action_value(action.value, name="temperature")
            return current is not None and current == target
        if action.operation is HomeOperation.TURN_OFF:
            return entity.state == "off"
        return False

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        if action.operation is HomeOperation.TURN_OFF:
            return StateConfirmation(frozenset({"off"}))
        return None


class MediaPlayerAdapter(DomainAdapter):
    domain = "media_player"
    active_states = frozenset({"playing"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.TURN_ON,
            HomeOperation.TURN_OFF,
            HomeOperation.PLAY,
            HomeOperation.PAUSE,
            HomeOperation.STOP,
        }
    )
    # Values follow Home Assistant's MediaPlayerEntityFeature contract. The
    # entity-specific bitmask remains the authority instead of domain defaults.
    _FEATURE_PAUSE = 1
    _FEATURE_TURN_ON = 128
    _FEATURE_TURN_OFF = 256
    _FEATURE_STOP = 4096
    _FEATURE_PLAY = 16384
    _feature_operations = (
        (_FEATURE_TURN_ON, HomeOperation.TURN_ON),
        (_FEATURE_TURN_OFF, HomeOperation.TURN_OFF),
        (_FEATURE_PLAY, HomeOperation.PLAY),
        (_FEATURE_PAUSE, HomeOperation.PAUSE),
        (_FEATURE_STOP, HomeOperation.STOP),
    )

    def operations(
        self,
        attributes: Mapping[str, object],
        service_schemas: Mapping[str, frozenset[str]],
    ) -> frozenset[HomeOperation]:
        value = attributes.get("supported_features")
        if isinstance(value, bool) or not isinstance(value, int):
            return self.default_operations
        return frozenset(
            {HomeOperation.QUERY}
            | {
                operation
                for feature, operation in self._feature_operations
                if value & feature
            }
        )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        service = {
            HomeOperation.TURN_ON: "turn_on",
            HomeOperation.TURN_OFF: "turn_off",
            HomeOperation.PLAY: "media_play",
            HomeOperation.PAUSE: "media_pause",
            HomeOperation.STOP: "media_stop",
        }.get(action.operation)
        if service is None:
            raise ValueError(
                f"{action.operation.value} is unsupported for domain: {self.domain}"
            )
        return ServiceCall(service, {})

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        return state_matches(
            entity.state,
            self._target_states(action, confirmation=False),
        )

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        return _state_confirmation(self._target_states(action, confirmation=True))

    @staticmethod
    def _target_states(
        action: HomeAction,
        *,
        confirmation: bool,
    ) -> frozenset[str] | None:
        if action.operation is HomeOperation.TURN_ON:
            states = {"on", "paused", "playing"}
            if confirmation:
                states.add("idle")
            return frozenset(states)
        return {
            HomeOperation.TURN_OFF: frozenset({"off"}),
            HomeOperation.PLAY: frozenset({"playing"}),
            HomeOperation.PAUSE: frozenset({"paused"}),
        }.get(action.operation)


class CoverAdapter(DomainAdapter):
    domain = "cover"
    active_states = frozenset({"open", "opening", "closing"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.OPEN,
            HomeOperation.CLOSE,
            HomeOperation.STOP,
            HomeOperation.SET_POSITION,
        }
    )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.SET_POSITION:
            return ServiceCall(
                "set_cover_position",
                {"position": percentage(action.value)},
            )
        service = {
            HomeOperation.OPEN: "open_cover",
            HomeOperation.CLOSE: "close_cover",
            HomeOperation.STOP: "stop_cover",
        }.get(action.operation)
        if service is None:
            raise ValueError(
                f"{action.operation.value} is unsupported for domain: {self.domain}"
            )
        return ServiceCall(service, {})

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        if action.operation is HomeOperation.SET_POSITION:
            current = numeric_value(entity.attributes.get("current_position"))
            return current is not None and current == float(percentage(action.value))
        target_position = self._target_position(action)
        current_position = numeric_value(entity.attributes.get("current_position"))
        if target_position is not None and current_position is not None:
            return current_position == float(target_position)
        return state_matches(entity.state, self._target_states(action))

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        target_position = self._target_position(action)
        current_position = numeric_value(entity.attributes.get("current_position"))
        if target_position is not None and current_position is not None:
            return CoverMovementConfirmation(current_position, target_position)
        return _state_confirmation(self._target_states(action))

    @staticmethod
    def _target_position(action: HomeAction) -> int | None:
        if action.operation is HomeOperation.OPEN:
            return 100
        if action.operation is HomeOperation.CLOSE:
            return 0
        return None

    @staticmethod
    def _target_states(action: HomeAction) -> frozenset[str] | None:
        return {
            HomeOperation.OPEN: frozenset({"open"}),
            HomeOperation.CLOSE: frozenset({"closed"}),
        }.get(action.operation)


class FanAdapter(DomainAdapter):
    domain = "fan"
    active_states = frozenset({"on"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.TURN_ON,
            HomeOperation.TURN_OFF,
            HomeOperation.SET_PERCENTAGE,
        }
    )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.SET_PERCENTAGE:
            return ServiceCall(
                "set_percentage",
                {"percentage": percentage(action.value)},
            )
        return _power_service_call(action, self.domain)

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        if action.operation is HomeOperation.SET_PERCENTAGE:
            current = numeric_value(entity.attributes.get("percentage"))
            return current is not None and current == float(percentage(action.value))
        return state_matches(entity.state, _power_target(action, entity.state))

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        return _state_confirmation(_power_target(action, entity.state))


class LockAdapter(DomainAdapter):
    domain = "lock"
    interesting_states = frozenset({"unlocked"})
    abnormal_states = frozenset({"jammed"})
    default_operations = frozenset(
        {HomeOperation.QUERY, HomeOperation.LOCK, HomeOperation.UNLOCK}
    )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        service = {
            HomeOperation.LOCK: "lock",
            HomeOperation.UNLOCK: "unlock",
        }.get(action.operation)
        if service is None:
            raise ValueError(
                f"{action.operation.value} is unsupported for domain: {self.domain}"
            )
        return ServiceCall(service, {})

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        return state_matches(entity.state, self._target_states(action))

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        return _state_confirmation(self._target_states(action))

    @staticmethod
    def _target_states(action: HomeAction) -> frozenset[str] | None:
        return {
            HomeOperation.LOCK: frozenset({"locked"}),
            HomeOperation.UNLOCK: frozenset({"unlocked"}),
        }.get(action.operation)


class VacuumAdapter(DomainAdapter):
    domain = "vacuum"
    active_states = frozenset({"cleaning", "returning"})
    abnormal_states = frozenset({"error"})
    default_operations = frozenset(
        {
            HomeOperation.QUERY,
            HomeOperation.START,
            HomeOperation.STOP,
            HomeOperation.RETURN_TO_BASE,
        }
    )

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        service = {
            HomeOperation.START: "start",
            HomeOperation.STOP: "stop",
            HomeOperation.RETURN_TO_BASE: "return_to_base",
        }.get(action.operation)
        if service is None:
            raise ValueError(
                f"{action.operation.value} is unsupported for domain: {self.domain}"
            )
        return ServiceCall(service, {})

    def target_is_satisfied(self, action: HomeAction, entity: HomeEntity) -> bool:
        expected = self._target_states(action)
        return state_matches(entity.state, expected)

    def confirmation(self, action: HomeAction, entity: HomeEntity) -> Confirmation:
        return _state_confirmation(self._target_states(action))

    @staticmethod
    def _target_states(action: HomeAction) -> frozenset[str] | None:
        if action.operation is HomeOperation.RETURN_TO_BASE:
            return frozenset({"returning", "docked"})
        return None


class SceneAdapter(DomainAdapter):
    domain = "scene"
    default_operations = frozenset({HomeOperation.ACTIVATE})

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.ACTIVATE:
            return ServiceCall("turn_on", {})
        return super().service_call(action, entity)


class ScriptAdapter(SceneAdapter):
    domain = "script"


class ButtonAdapter(DomainAdapter):
    domain = "button"
    default_operations = frozenset({HomeOperation.ACTIVATE})

    def service_call(self, action: HomeAction, entity: HomeEntity) -> ServiceCall:
        if action.operation is HomeOperation.ACTIVATE:
            return ServiceCall("press", {})
        return super().service_call(action, entity)


def _power_service_call(action: HomeAction, domain: str) -> ServiceCall:
    service = {
        HomeOperation.TURN_ON: "turn_on",
        HomeOperation.TURN_OFF: "turn_off",
        HomeOperation.TOGGLE: "toggle",
    }.get(action.operation)
    if service is None:
        raise ValueError(
            f"{action.operation.value} is unsupported for domain: {domain}"
        )
    return ServiceCall(service, {})


def _power_target(
    action: HomeAction,
    current_state: str,
) -> frozenset[str] | None:
    if action.operation is HomeOperation.TURN_ON:
        return frozenset({"on"})
    if action.operation is HomeOperation.TURN_OFF:
        return frozenset({"off"})
    if action.operation is HomeOperation.TOGGLE:
        return toggle_target(current_state)
    return None


def _state_confirmation(states: frozenset[str] | None) -> Confirmation:
    return StateConfirmation(states) if states is not None else None


def _supports_color_temperature(
    attributes: Mapping[str, object],
    turn_on_fields: frozenset[str],
) -> bool:
    modes = attributes.get("supported_color_modes")
    supports_mode = (
        isinstance(modes, Sequence)
        and not isinstance(modes, (str, bytes))
        and "color_temp" in modes
    )
    has_kelvin_range = all(
        numeric_value(attributes.get(name)) is not None
        for name in ("min_color_temp_kelvin", "max_color_temp_kelvin")
    )
    has_mired_range = all(
        numeric_value(attributes.get(name)) is not None
        for name in ("min_mireds", "max_mireds")
    )
    return (
        supports_mode
        and bool({"color_temp_kelvin", "color_temp"} & turn_on_fields)
        and (has_kelvin_range or has_mired_range)
    )


def _color_temperature_kelvin(value: object, entity: HomeEntity) -> int:
    minimum, maximum = _color_temperature_range_kelvin(entity)
    if minimum > maximum:
        raise ValueError("invalid light color temperature range")
    if isinstance(value, str):
        target = {
            "warm": minimum,
            "neutral": (minimum + maximum) / 2,
            "cool": maximum,
        }.get(value.strip().casefold())
        if target is None:
            raise ValueError("color temperature must be warm, neutral, cool, or Kelvin")
    else:
        target = numeric_action_value(value, name="color temperature")
    return round(min(max(target, minimum), maximum))


def _color_temperature_range_kelvin(entity: HomeEntity) -> tuple[float, float]:
    minimum = numeric_value(entity.attributes.get("min_color_temp_kelvin"))
    maximum = numeric_value(entity.attributes.get("max_color_temp_kelvin"))
    if minimum is not None and maximum is not None:
        return minimum, maximum
    min_mireds = _required_numeric_attribute(entity, "min_mireds")
    max_mireds = _required_numeric_attribute(entity, "max_mireds")
    return 1_000_000 / max_mireds, 1_000_000 / min_mireds


def _required_numeric_attribute(entity: HomeEntity, name: str) -> float:
    value = numeric_value(entity.attributes.get(name))
    if value is None:
        raise ValueError(f"light is missing numeric {name}")
    return value
