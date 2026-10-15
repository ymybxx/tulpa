"""Shared semantic history levels for scheduling, storage, and presentation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SnapshotLevel:
    period_seconds: int
    child_period_seconds: int | None
    completed_field: str
    failed_field: str

    def __post_init__(self) -> None:
        if self.period_seconds <= 0:
            raise ValueError("snapshot period must be positive")
        child = self.child_period_seconds
        if child is not None and (
            child <= 0 or child >= self.period_seconds or self.period_seconds % child
        ):
            raise ValueError("snapshot parent must contain whole shorter children")

    @property
    def observation_type(self) -> str:
        return f"semantic_snapshot_{self.period_seconds}s"

    @property
    def child_count(self) -> int:
        child = self.child_period_seconds
        return self.period_seconds // child if child is not None else 0


SNAPSHOT_LEVELS = (
    SnapshotLevel(5, None, "completedWindows", "failedWindows"),
    SnapshotLevel(10, None, "completedWindows", "failedWindows"),
    SnapshotLevel(60, 10, "completedMinutes", "failedMinutes"),
    SnapshotLevel(300, 60, "completedFiveMinutes", "failedFiveMinutes"),
    SnapshotLevel(1800, 300, "completedHalfHours", "failedHalfHours"),
)


def snapshot_levels() -> tuple[SnapshotLevel, ...]:
    """Return the configured levels, from finest to coarsest."""
    return SNAPSHOT_LEVELS


def level_for_period(period_seconds: int) -> SnapshotLevel:
    for level in snapshot_levels():
        if level.period_seconds == period_seconds:
            return level
    raise ValueError(f"unsupported semantic snapshot period: {period_seconds}")


def parent_level_for_period(child_period_seconds: int) -> SnapshotLevel | None:
    level_for_period(child_period_seconds)
    return next(
        (
            level
            for level in snapshot_levels()
            if level.child_period_seconds == child_period_seconds
        ),
        None,
    )
