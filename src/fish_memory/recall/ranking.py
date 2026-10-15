from collections.abc import Sequence
from math import isfinite


def dominant_gap_size(scores: Sequence[float]) -> int:
    count = len(scores)
    if count < 4 or any(not isfinite(score) for score in scores):
        return count
    if any(scores[index] < scores[index + 1] for index in range(count - 1)):
        raise ValueError("dominant gap scores must be descending")

    gaps = [scores[index] - scores[index + 1] for index in range(count - 1)]
    boundary = max(range(len(gaps)), key=gaps.__getitem__)
    gap = gaps[boundary]
    upper_range = scores[0] - scores[boundary]
    lower_range = scores[boundary + 1] - scores[-1]
    if gap > upper_range + lower_range:
        return boundary + 1
    return count
