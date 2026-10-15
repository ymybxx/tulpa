from fish_memory.background.snapshot_orchestrator import (
    SnapshotDerivationError,
    primary_orchestration_exception,
)


def test_blocking_error_wins_over_skippable_error_in_exception_group() -> None:
    """Catches one parallel provider error hiding a consistency failure."""
    skippable = SnapshotDerivationError(
        stage="preupdate",
        cause=ValueError("malformed provider output"),
    )
    blocking = RuntimeError("database evidence load failed")

    primary = primary_orchestration_exception(
        ExceptionGroup("parallel failures", (skippable, blocking))
    )

    assert primary is blocking


def test_first_derivation_error_is_used_when_every_parallel_error_is_skippable() -> (
    None
):
    """Catches all-provider failures being mislabeled as an infrastructure error."""
    first = SnapshotDerivationError(
        stage="preupdate",
        cause=ValueError("bad preupdate"),
    )
    second = SnapshotDerivationError(
        stage="entity_resolution",
        cause=ValueError("bad entity output"),
    )

    primary = primary_orchestration_exception(
        ExceptionGroup("parallel failures", (first, second))
    )

    assert primary is first
