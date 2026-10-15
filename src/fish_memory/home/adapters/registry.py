from __future__ import annotations

from fish_memory.home.adapters.base import DomainAdapter
from fish_memory.home.adapters.domains import (
    ButtonAdapter,
    ClimateAdapter,
    CoverAdapter,
    FanAdapter,
    LightAdapter,
    LockAdapter,
    MediaPlayerAdapter,
    SceneAdapter,
    ScriptAdapter,
    SwitchAdapter,
    VacuumAdapter,
)

_ADAPTERS: dict[str, DomainAdapter] = {
    adapter.domain: adapter
    for adapter in (
        LightAdapter(),
        SwitchAdapter(),
        ClimateAdapter(),
        MediaPlayerAdapter(),
        CoverAdapter(),
        FanAdapter(),
        LockAdapter(),
        VacuumAdapter(),
        SceneAdapter(),
        ScriptAdapter(),
        ButtonAdapter(),
    )
}


def adapter_for_domain(domain: str) -> DomainAdapter:
    try:
        return _ADAPTERS[domain]
    except KeyError as error:
        raise ValueError(f"unsupported Home Assistant domain: {domain}") from error


def supported_domains() -> frozenset[str]:
    return frozenset(_ADAPTERS)
