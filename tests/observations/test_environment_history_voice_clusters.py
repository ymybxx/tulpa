from itertools import permutations

import numpy as np
import pytest

from fish_memory.observations.environment_history_voice_clusters import (
    VoiceClusterBank,
    VoiceEvidence,
    _assignment,
    average_link_groups,
)


def evidence(start, vector):
    value = np.asarray(vector, dtype=np.float32)
    value /= np.linalg.norm(value)
    return VoiceEvidence(start, start + 16000, value, start)


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (3, 3), (4, 2)])
def test_identity_assignment_finds_global_optimum_with_unmatched_groups(shape):
    scores = np.random.default_rng(42).integers(-2, 10, size=shape).astype(float)
    matches = _assignment(scores)
    assert len({r for r, _ in matches}) == len(matches)
    assert len({c for _, c in matches}) == len(matches)
    extended = np.pad(scores, ((0, 0), (0, shape[0])))
    best = max(
        sum(extended[r, c] for r, c in enumerate(choice))
        for choice in permutations(range(extended.shape[1]), shape[0])
    )
    assert sum(scores[r, c] for r, c in matches) == best


def test_average_link_uses_all_evidence_not_a_greedy_first_voice_anchor():
    vectors = np.asarray([[1, 0], [0.95, 0.1], [0, 1], [0.1, 0.95]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = average_link_groups(vectors)
    assert {frozenset(g) for g in groups} == {frozenset((0, 1)), frozenset((2, 3))}


def test_later_short_and_long_evidence_keep_one_identity_and_correct_old_labels():
    bank = VoiceClusterBank()
    first = evidence(0, [1, 0])
    left = bank.reconcile((first,))[first.key]
    data = (
        first,
        evidence(20000, [0, 1]),
        evidence(40000, [0.95, 0.1]),
        evidence(60000, [0.1, 0.95]),
    )
    labels = bank.reconcile(data)
    assert labels[first.key] == labels[data[2].key] == left
    assert labels[data[1].key] == labels[data[3].key] != left
    assert bank.copy().reconcile(tuple(reversed(data))) == labels


def test_eviction_matches_saved_voice_but_never_recycles_another_person_id():
    bank = VoiceClusterBank(capacity=1)
    original = tuple(evidence(i * 20000, [1, 0]) for i in range(4))
    identity = bank.reconcile(original)[original[0].key]
    similar = tuple(evidence(1_000_000 + i * 20000, [1, 0.02]) for i in range(4))
    assert set(bank.reconcile(similar).values()) == {identity}
    changed = tuple(evidence(2_000_000 + i * 20000, [0, 1]) for i in range(4))
    replacement = set(bank.reconcile(changed).values())
    assert identity not in replacement
    assert len(bank.profiles) == 1


def test_capacity_leaves_excess_evidence_unassigned_instead_of_merging_people():
    bank = VoiceClusterBank(capacity=1)
    first, second = evidence(0, [1, 0]), evidence(20000, [0, 1])
    labels = bank.reconcile((first, second))
    assert len(labels) == 1
    assert len(set(labels.values())) == 1
