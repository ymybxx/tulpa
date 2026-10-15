from types import MappingProxyType

from fish_memory.home.adapters import (
    CoverMovementConfirmation,
    adapter_for_domain,
    supported_domains,
)
from fish_memory.home.models import HomeAction, HomeEntity, HomeOperation


def _cover(*, state: str = "open", position: int = 1) -> HomeEntity:
    return HomeEntity(
        entity_id="cover.bedroom",
        domain="cover",
        state=state,
        attributes=MappingProxyType({"current_position": position}),
        aliases=("卧室窗帘",),
        operations=frozenset(
            {
                HomeOperation.QUERY,
                HomeOperation.OPEN,
                HomeOperation.CLOSE,
                HomeOperation.STOP,
                HomeOperation.SET_POSITION,
            }
        ),
        service_schemas=MappingProxyType(
            {
                "open_cover": frozenset(),
                "close_cover": frozenset(),
            }
        ),
    )


def test_registry_has_one_adapter_for_every_supported_domain() -> None:
    domains = {
        "light",
        "switch",
        "climate",
        "media_player",
        "cover",
        "fan",
        "lock",
        "vacuum",
        "scene",
        "script",
        "button",
    }

    assert supported_domains() == frozenset(domains)
    assert {adapter_for_domain(domain).domain for domain in domains} == domains


def test_cover_adapter_owns_service_target_and_confirmation_rules() -> None:
    adapter = adapter_for_domain("cover")
    entity = _cover()
    action = HomeAction(entity.entity_id, HomeOperation.OPEN)

    call = adapter.service_call(action, entity)
    confirmation = adapter.confirmation(action, entity)

    assert call.service == "open_cover"
    assert call.data == {}
    assert adapter.target_is_satisfied(action, entity) is False
    assert confirmation == CoverMovementConfirmation(
        initial_position=1.0,
        target_position=100,
    )


def test_cover_adapter_only_treats_full_position_as_open() -> None:
    adapter = adapter_for_domain("cover")
    entity = _cover(position=100)
    action = HomeAction(entity.entity_id, HomeOperation.OPEN)

    assert adapter.target_is_satisfied(action, entity) is True
