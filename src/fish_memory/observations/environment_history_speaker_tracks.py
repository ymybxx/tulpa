"""Bounded provisional voices, independent of recognition against old voices.

Each anchor comes from one independently embedded, clean long turn. Model-local
channels only constrain associations; they never establish them. Short turns
can match an anchor but cannot move it or contribute to the accumulated voice.
Thresholds are conservative initial values, pending paired real-audio tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

_MAX_MEMBERS = 4
_MAX_ABSENCE_WINDOWS = 6
_LONG_SAMPLES = 19_200


def reference_score(
    vector: NDArray[np.float32],
    anchor: NDArray[np.float32],
    members: tuple[NDArray[np.float32], ...],
) -> float:
    """The original reference or two independent accepted turns must agree.

    A single unusual member cannot attract another voice, and an old reference
    cannot veto two later, independently verified examples of the same voice.
    Callers admit members only when they also match the immutable anchor.
    """
    support = sorted((float(np.dot(vector, item)) for item in members), reverse=True)
    return max(float(np.dot(vector, anchor)), support[1] if len(support) > 1 else -1.0)


@dataclass(frozen=True, slots=True)
class TrackMatch:
    track_id: str | None = None
    reason: str = "short_evidence"


@dataclass(frozen=True, slots=True)
class _Track:
    anchor: NDArray[np.float32]
    members: tuple[NDArray[np.float32], ...]
    weights: tuple[float, ...]
    centroid: NDArray[np.float32]
    identity: str | None
    last_window: int


class SpeakerTrackTable:
    """Immutable records make shallow snapshots safe for the caller's rollback."""

    def __init__(self, namespace: str, capacity: int, threshold: float, margin: float):
        self.namespace = namespace
        self.capacity = min(64, capacity * 2)
        self.threshold = max(0.70, threshold)
        self.member_threshold = threshold
        self.margin = max(0.10, margin)
        self.tracks: dict[str, _Track] = {}
        self.canonical: dict[str, str] = {}
        self.next_id = 1

    def clear(self, *, keep_canonical: bool = False) -> None:
        self.tracks.clear()
        if not keep_canonical:
            self.canonical.clear()

    def snapshot(self) -> tuple[dict[str, _Track], dict[str, str], int]:
        return self.tracks.copy(), self.canonical.copy(), self.next_id

    def restore(self, state: tuple[dict[str, _Track], dict[str, str], int]) -> None:
        self.tracks, self.canonical, self.next_id = state

    def _bind(self, identity: str | None, track_id: str) -> None:
        if identity is None:
            return
        previous = self.canonical.get(identity)
        if previous is not None and previous != track_id:
            return
        self.canonical.pop(identity, None)
        self.canonical[identity] = track_id
        while len(self.canonical) > self.capacity:
            del self.canonical[next(iter(self.canonical))]

    @staticmethod
    def _score(vector: NDArray[np.float32], track: _Track) -> float:
        return reference_score(vector, track.anchor, track.members)

    def assign(
        self,
        vectors: dict[int, NDArray[np.float32]],
        clean_samples: dict[int, int],
        channels: dict[int, int],
        identities: dict[int, str],
        *,
        window_index: int,
    ) -> dict[int, TrackMatch]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if window_index - value.last_window <= _MAX_ABSENCE_WINDOWS
        }
        matches: dict[int, TrackMatch] = {}
        occupied: dict[str, int] = {}
        # Establish the clearest independent anchor first, including when the
        # first short utterance in time precedes the useful long utterance.
        for island in sorted(vectors, key=lambda item: (-clean_samples[item], item)):
            vector, identity = vectors[island], identities.get(island)
            long = clean_samples[island] >= _LONG_SAMPLES
            threshold = max(self.threshold, 0.72 if not long else 0.70)
            margin = max(self.margin, 0.12 if not long else 0.10)
            scores = sorted(
                (
                    (self._score(vector, track), key)
                    for key, track in self.tracks.items()
                ),
                reverse=True,
            )
            reason = "new_anchor" if long else "match_below_threshold"
            selected = None
            canonical = self.canonical.get(identity) if identity else None
            if canonical and (
                canonical not in occupied or occupied[canonical] == channels[island]
            ):
                # Global matching already passed its own thresholds and local
                # cannot-link checks. A stricter provisional gate must not split
                # that confirmed identity into additional public voice labels.
                if canonical in self.tracks:
                    selected = canonical
                    reason = "matched_identity"
                elif not long:
                    # The voice table retains the authoritative reference even
                    # when a provisional anchor expired/was evicted. Reuse only
                    # the label; weak evidence still cannot seed a new anchor.
                    self._bind(identity, canonical)
                    occupied[canonical] = channels[island]
                    matches[island] = TrackMatch(canonical, "matched_identity")
                    continue
            if (
                selected is None
                and canonical is None
                and scores
                and scores[0][0] >= threshold
            ):
                score, key = scores[0]
                track = self.tracks[key]
                competitors = [
                    self._score(other_vector, track)
                    for other, other_vector in vectors.items()
                    if channels[other] != channels[island]
                    and (not long or clean_samples[other] >= _LONG_SAMPLES)
                ]
                if len(scores) > 1 and score - scores[1][0] < margin:
                    reason = "match_ambiguous"
                elif competitors and score - max(competitors) < margin:
                    # Duration chooses an independent seed; it cannot decide
                    # which of two competing model-local people owns an old
                    # track. A closer rival wins and a near tie keeps both out.
                    reason = "local_conflict"
                elif key in occupied and occupied[key] != channels[island]:
                    reason = "local_conflict"
                elif identity and track.identity and identity != track.identity:
                    reason = "identity_conflict"
                else:
                    selected = key
            if selected is not None:
                track = self.tracks.pop(selected)
                members, weights, centroid = (
                    track.members,
                    track.weights,
                    track.centroid,
                )
                # An independently long, very close member can strengthen the
                # aggregate. Weak short recognitions never become new evidence.
                if (
                    long
                    and vector is not track.anchor
                    and all(vector is not member for member in track.members)
                    and float(np.dot(vector, track.anchor))
                    >= (
                        self.member_threshold
                        if identity is not None
                        else self.threshold
                    )
                ):
                    members = (*members[-(_MAX_MEMBERS - 1) :], vector)
                    weights = (
                        *weights[-(_MAX_MEMBERS - 1) :],
                        min(clean_samples[island], 80_000) / 16_000,
                    )
                    combined = sum(
                        (
                            member * weight
                            for member, weight in zip(members, weights, strict=True)
                        ),
                        track.anchor * 2,
                    )
                    centroid = combined / np.linalg.norm(combined)
                self.tracks[selected] = replace(
                    track,
                    members=members,
                    weights=weights,
                    centroid=centroid,
                    identity=track.identity or identity,
                    last_window=window_index,
                )
                occupied[selected] = channels[island]
                self._bind(identity, selected)
                matches[island] = TrackMatch(
                    selected,
                    "matched_identity"
                    if reason == "matched_identity"
                    else "matched_anchor",
                )
                continue
            if not long:
                matches[island] = TrackMatch(reason=reason)
                continue
            # Failure to recognize an old voice is not evidence of a new
            # person. The caller confirms new identities from independent long
            # turns before they acquire a public track number.
            if identity is None:
                matches[island] = TrackMatch(reason="pending_identity")
                continue
            if len(self.tracks) >= self.capacity:
                victim = next((key for key in self.tracks if key not in occupied), None)
                if victim is None:
                    matches[island] = TrackMatch(reason="capacity")
                    continue
                del self.tracks[victim]
            if canonical and canonical not in self.tracks and canonical not in occupied:
                key = canonical
                reason = "reestablished_identity"
            else:
                key = f"track:{self.namespace}:{self.next_id}"
                self.next_id += 1
            self.tracks[key] = _Track(vector, (), (), vector, identity, window_index)
            self._bind(identity, key)
            occupied[key] = channels[island]
            matches[island] = TrackMatch(key, reason)
        return matches

    def merge_identities(self, old: str, retained: str) -> tuple[str, str] | None:
        """Apply the caller's independently verified, cannot-link checked merge."""
        old_key, new_key = self.canonical.get(old), self.canonical.get(retained)
        if old_key is None:
            return None
        if new_key is None:
            self._bind(retained, old_key)
            new_key = old_key
        self.canonical[old] = new_key
        if new_key in self.tracks:
            self.tracks[new_key] = replace(self.tracks[new_key], identity=retained)
        if old_key == new_key:
            return None
        self.tracks.pop(old_key, None)
        return old_key, new_key

    def install_identity(
        self,
        identity: str,
        anchor: NDArray[np.float32],
        members: tuple[NDArray[np.float32], ...],
        *,
        window_index: int,
    ) -> None:
        """Use the caller's confirmed references, including corroborated shorts."""
        key = self.canonical.get(identity)
        if key is not None and key in self.tracks:
            return
        if len(self.tracks) >= self.capacity:
            del self.tracks[next(iter(self.tracks))]
        if key is None:
            key = f"track:{self.namespace}:{self.next_id}"
            self.next_id += 1
        accepted = members[-_MAX_MEMBERS:]
        self.tracks[key] = _Track(
            anchor, accepted, (1.0,) * len(accepted), anchor, identity, window_index
        )
        self._bind(identity, key)

    def inherit(
        self,
        track_id: str,
        *,
        island: int,
        channels: dict[int, int],
        matches: dict[int, TrackMatch],
        vector: NDArray[np.float32] | None,
        identity: str | None,
        window_index: int,
    ) -> TrackMatch:
        if any(
            match.track_id == track_id and channels[index] != channels[island]
            for index, match in matches.items()
        ):
            return TrackMatch(reason="local_conflict")
        track = self.tracks.get(track_id)
        if track is None:
            if identity and self.canonical.get(identity) == track_id:
                # The caller has independently confirmed this global identity
                # at the seam; no weak audio is added to any template here.
                return TrackMatch(track_id, "matched_identity")
            return TrackMatch(reason="track_unavailable")
        if identity and track.identity and identity != track.identity:
            return TrackMatch(reason="identity_conflict")
        if vector is not None:
            score = self._score(vector, track)
            alternatives = [
                self._score(vector, value)
                for key, value in self.tracks.items()
                if key != track_id
            ]
            if score < self.threshold or (
                alternatives and score - max(alternatives) < self.margin
            ):
                return TrackMatch(reason="embedding_conflict")
        self.tracks.pop(track_id)
        self.tracks[track_id] = replace(track, last_window=window_index)
        return TrackMatch(track_id, "boundary_continuity")
