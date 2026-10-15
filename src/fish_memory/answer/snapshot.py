from datetime import datetime, timedelta

from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
)


def render_dynamic_snapshot(
    snapshot: NormalizedDynamicSnapshot,
    current_time: datetime,
    max_age: timedelta,
) -> str:
    active_items = snapshot.active_items(current_time, max_age)
    if not active_items:
        return ""
    return "\n".join(
        (
            "<dynamic_memory>",
            *(item.summary for item in active_items),
            "</dynamic_memory>",
        )
    )


def render_static_profile(profile: NormalizedStaticProfile) -> str:
    active_items = profile.active_items()
    if not active_items:
        return ""
    return "\n".join(
        (
            "<static_memory>",
            *(item.summary for item in active_items),
            "</static_memory>",
        )
    )
