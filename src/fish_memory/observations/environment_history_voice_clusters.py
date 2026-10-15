"""Cluster cached acoustic evidence and retain identities across updates.

Average-link cosine clustering follows the approach used by 3D-Speaker's AHC
backend. It operates on bounded, already extracted embeddings, never on old PCM.
No character count, transcript, name, or story annotation is supplied here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
EvidenceKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class VoiceEvidence:
    start: int
    end: int
    vector: Vector
    run_start: int

    @property
    def key(self) -> EvidenceKey:
        return self.start, self.end


def average_link_groups(
    vectors: Vector, minimum_similarity: float = 0.4
) -> list[list[int]]:
    if not len(vectors):
        return []
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    active = np.ones(len(vectors), dtype=bool)
    weights = np.ones(len(vectors), dtype=np.float64)
    groups = [[i] for i in range(len(vectors))]
    while active.sum() > 1:
        first, second = np.unravel_index(np.argmax(similarities), similarities.shape)
        if similarities[first, second] < minimum_similarity:
            break
        keep = np.flatnonzero(
            active
            & (np.arange(len(active)) != first)
            & (np.arange(len(active)) != second)
        )
        values = (
            weights[first] * similarities[first, keep]
            + weights[second] * similarities[second, keep]
        ) / (weights[first] + weights[second])
        similarities[first, keep] = values
        similarities[keep, first] = values
        similarities[second, :] = similarities[:, second] = -np.inf
        active[second] = False
        weights[first] += weights[second]
        groups[first].extend(groups[second])
    return [group for i, group in enumerate(groups) if active[i]]


def _center(vectors: Vector) -> Vector:
    center = vectors.mean(axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-8)


def _assignment(scores: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Rectangular maximum-weight assignment, with private unmatched columns."""
    rows, columns = scores.shape
    if not rows or not columns:
        return []
    costs = -np.pad(scores, ((0, 0), (0, rows)))
    n, m = costs.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    matched, path = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        matched[0] = row
        column = 0
        distances = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[column] = True
            current = matched[column]
            available = np.flatnonzero(~used[1:]) + 1
            reduced = costs[current - 1, available - 1] - u[current] - v[available]
            better = reduced < distances[available]
            distances[available[better]] = reduced[better]
            path[available[better]] = column
            following = available[np.argmin(distances[available])]
            delta = distances[following]
            u[matched[used]] += delta
            v[used] -= delta
            distances[~used] -= delta
            column = following
            if matched[column] == 0:
                break
        while column:
            previous = path[column]
            matched[column] = matched[previous]
            column = previous
    return [
        (matched[column] - 1, column - 1)
        for column in range(1, columns + 1)
        if matched[column] and scores[matched[column] - 1, column - 1] > 0
    ]


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    center: Vector
    last_seen: int


class VoiceClusterBank:
    def __init__(
        self,
        capacity: int = 16,
        *,
        match_threshold: float = 0.65,
        match_margin: float = 0.10,
    ) -> None:
        self.capacity = capacity
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.labels: dict[EvidenceKey, int] = {}
        self.profiles: dict[int, VoiceProfile] = {}
        self.confirmed_ids: set[int] = set()
        self.next_identity = 1

    def copy(self) -> VoiceClusterBank:
        copied = VoiceClusterBank(
            self.capacity,
            match_threshold=self.match_threshold,
            match_margin=self.match_margin,
        )
        copied.labels = self.labels.copy()
        copied.profiles = self.profiles.copy()
        copied.confirmed_ids = self.confirmed_ids.copy()
        copied.next_identity = self.next_identity
        return copied

    def reconcile(self, evidence: tuple[VoiceEvidence, ...]) -> dict[EvidenceKey, int]:
        if not evidence:
            self.labels = {}
            return {}
        vectors = np.stack([item.vector for item in evidence])
        groups = average_link_groups(vectors)
        centers = [_center(vectors[group]) for group in groups]
        substantial = [i for i, group in enumerate(groups) if len(group) >= 4]
        retained = []
        for index, group in enumerate(groups):
            if index not in substantial and substantial:
                ranked = sorted(
                    (
                        (float(centers[index] @ centers[other]), other)
                        for other in substantial
                    ),
                    reverse=True,
                )
                if ranked[0][0] >= 0.5 and (
                    len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05
                ):
                    groups[ranked[0][1]].extend(group)
                    continue
            retained.append(index)
        groups = [groups[i] for i in retained]
        # Capacity is a memory bound, not an oracle for the number of people.
        # Weak excess evidence remains unassigned rather than being forced into
        # a known person's cluster.
        groups = sorted(
            groups, key=lambda g: -sum(evidence[i].end - evidence[i].start for i in g)
        )[: self.capacity]
        groups.sort(key=lambda g: min(evidence[i].start for i in g))
        centers = [_center(vectors[group]) for group in groups]
        identities = sorted(self.profiles.keys() | set(self.labels.values()))
        scores = np.zeros((len(groups), len(identities)))
        for row, group in enumerate(groups):
            overlaps = Counter(
                self.labels[item.key]
                for i in group
                if (item := evidence[i]).key in self.labels
            )
            if overlaps:
                for column, identity in enumerate(identities):
                    scores[row, column] = overlaps[identity] * 100.0
            else:
                candidates = sorted(
                    (
                        (float(centers[row] @ profile.center), identity)
                        for identity, profile in self.profiles.items()
                    ),
                    reverse=True,
                )
                if (
                    candidates
                    and candidates[0][0] >= self.match_threshold
                    and (
                        len(candidates) == 1
                        or candidates[0][0] - candidates[1][0] >= self.match_margin
                    )
                ):
                    scores[row, identities.index(candidates[0][1])] = candidates[0][0]
        assigned = {row: identities[column] for row, column in _assignment(scores)}
        labels: dict[EvidenceKey, int] = {}
        for index, group in enumerate(groups):
            identity = assigned.get(index)
            if identity is None:
                identity = self.next_identity
                self.next_identity += 1
            for item in group:
                labels[evidence[item].key] = identity
            covered_until = -1
            unique_samples = 0
            for item in sorted((evidence[i] for i in group), key=lambda x: x.key):
                unique_samples += max(0, item.end - max(item.start, covered_until))
                covered_until = max(covered_until, item.end)
            if unique_samples >= 19_200:
                # Count actual unique audio, not overlapping crops or repeated
                # padding. A weak tail remains anonymous until later acoustic
                # evidence links it to a sufficiently supported voice.
                self.confirmed_ids.add(identity)
                self.profiles[identity] = VoiceProfile(
                    centers[index], max(evidence[i].end for i in group)
                )
        self.profiles = dict(
            sorted(self.profiles.items(), key=lambda item: -item[1].last_seen)[
                : self.capacity
            ]
        )
        self.labels = labels
        self.confirmed_ids.intersection_update(
            self.profiles.keys() | set(labels.values())
        )
        return labels.copy()
