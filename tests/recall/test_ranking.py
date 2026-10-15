import pytest

from fish_memory.recall.ranking import dominant_gap_size


def test_dominant_gap_separates_two_compact_score_bands() -> None:
    assert dominant_gap_size((0.92, 0.91, 0.90, 0.62, 0.61, 0.60)) == 3


@pytest.mark.parametrize(
    "scores",
    [
        (),
        (0.9,),
        (0.9, 0.8),
        (0.9, 0.89, 0.88, 0.87, 0.86),
    ],
)
def test_dominant_gap_keeps_all_without_a_dominating_separation(
    scores: tuple[float, ...],
) -> None:
    assert dominant_gap_size(scores) == len(scores)


@pytest.mark.parametrize(
    "scores",
    [
        (0.9, 0.8, 0.7),
        (0.8, 0.6, 0.4),
    ],
)
def test_dominant_gap_keeps_all_when_fewer_than_four_scores(
    scores: tuple[float, ...],
) -> None:
    assert dominant_gap_size(scores) == len(scores)


def test_dominant_gap_keeps_near_flat_finite_scores() -> None:
    scores = (0.9, 0.8999900000000001, 0.89998, 0.89997)

    assert dominant_gap_size(scores) == len(scores)


def test_dominant_gap_keeps_exact_ties() -> None:
    scores = (0.8, 0.8, 0.8, 0.8)

    assert dominant_gap_size(scores) == len(scores)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_dominant_gap_keeps_all_with_a_nonfinite_score(nonfinite: float) -> None:
    scores = (0.9, nonfinite, 0.8, 0.7)

    assert dominant_gap_size(scores) == len(scores)


def test_dominant_gap_separates_genuinely_split_four_score_sequence() -> None:
    assert dominant_gap_size((0.95, 0.94, 0.60, 0.59)) == 2


def test_dominant_gap_rejects_non_descending_finite_scores() -> None:
    with pytest.raises(ValueError, match="dominant gap scores must be descending"):
        dominant_gap_size((0.9, 0.7, 0.8, 0.6))
