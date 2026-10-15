from __future__ import annotations

import numpy as np
import pytest

from fish_memory.observations.environment_history_speaker_tracks import (
    SpeakerTrackTable,
)


def _vector(index: int) -> np.ndarray:
    vector = np.zeros(192, dtype=np.float32)
    vector[index] = 1
    return vector


def _similarity(value: float, *, axis: int = 1) -> np.ndarray:
    return _vector(0) * value + _vector(axis) * np.sqrt(1 - value**2)


def _table(capacity: int = 16) -> SpeakerTrackTable:
    return SpeakerTrackTable("abc123", capacity, 0.65, 0.10)


def _assign(table: SpeakerTrackTable, vector: np.ndarray, index: int, samples=32_000):
    return table.assign({0: vector}, {0: samples}, {0: 0}, {}, window_index=index)[0]


def _seed(
    table: SpeakerTrackTable, vector: np.ndarray, index: int = 0, identity="voice-a"
):
    return table.assign(
        {0: vector}, {0: 32_000}, {0: 0}, {0: identity}, window_index=index
    )[0]


def test_short_voice_recognizes_fixed_anchor_without_mutating_any_voice_evidence():
    table = _table()
    key = _seed(table, _vector(0), 0).track_id
    assert key is not None
    before = table.tracks[key]
    match = _assign(table, _similarity(0.9), 1, samples=12_000)
    assert match.track_id == key
    after = table.tracks[key]
    assert after.anchor is before.anchor
    assert after.centroid is before.centroid
    assert after.members is before.members
    assert after.weights == ()


def test_accepted_long_members_accumulate_only_after_independent_consistency_check():
    table = _table()
    key = _seed(table, _vector(0), 0).track_id
    assert key is not None
    match = _assign(table, _similarity(0.9), 1)
    assert match.track_id == key
    track = table.tracks[key]
    assert len(track.members) == 1
    np.testing.assert_array_equal(track.anchor, _vector(0))
    assert 0 < track.centroid[1] < _similarity(0.9)[1]
    # This turn resembles the updated centroid much more than the immutable
    # anchor; a chain of drifting associations must not pull in a second voice.
    incompatible = _similarity(0.69)
    assert np.dot(incompatible, track.centroid) > 0.70
    changed = _assign(table, incompatible, 2)
    assert changed.track_id != key
    assert len(table.tracks[key].members) == 1


def test_member_conflict_cannot_be_hidden_by_similarity_to_anchor():
    table = _table()
    key = _seed(table, _vector(0), 0).track_id
    _assign(table, _similarity(0.81), 1)
    # One anomalous new reference must not attract a query that does not
    # independently match the original anchor.
    conflicting = _vector(0) * 0.55 + _vector(1) * np.sqrt(1 - 0.55**2)
    result = _assign(table, conflicting, 2)
    assert result.track_id != key


def test_same_model_channel_never_proves_two_conflicting_voices_are_one():
    table = _table()
    results = table.assign(
        {0: _vector(0), 1: _vector(1)},
        {0: 32_000, 1: 48_000},
        {0: 0, 1: 0},
        {0: "voice-a", 1: "voice-b"},
        window_index=0,
    )
    assert results[0].track_id and results[1].track_id
    assert results[0].track_id != results[1].track_id


def test_simultaneously_distinct_local_channels_cannot_share_track():
    table = _table()
    results = table.assign(
        {0: _vector(0), 1: _vector(0)},
        {0: 32_000, 1: 48_000},
        {0: 0, 1: 1},
        {0: "voice-a", 1: "voice-b"},
        window_index=0,
    )
    assert results[0].track_id != results[1].track_id
    assert results[0].reason in {"local_conflict", "pending_identity"}


def test_stronger_competing_local_voice_wins_even_when_it_speaks_for_less_time():
    table = _table()
    original = _seed(table, _vector(0), 0).track_id
    results = table.assign(
        {0: _similarity(0.75), 1: _vector(0)},
        {0: 64_000, 1: 32_000},
        {0: 0, 1: 1},
        {},
        window_index=1,
    )
    assert results[1].track_id == original
    assert results[0].track_id != original
    assert results[0].reason in {"local_conflict", "pending_identity"}


def test_tied_local_people_cannot_claim_previously_known_track():
    table = _table()
    original = _seed(table, _vector(0), 0).track_id
    results = table.assign(
        {0: _similarity(0.9, axis=1), 1: _similarity(0.9, axis=2)},
        {0: 64_000, 1: 32_000},
        {0: 0, 1: 1},
        {},
        window_index=1,
    )
    assert all(result.track_id != original for result in results.values())


def test_matching_historical_identity_does_not_renumber_track():
    table = _table()
    old = _seed(table, _vector(0), 0)
    result = table.assign(
        {0: _vector(0)}, {0: 32_000}, {0: 2}, {0: "voice-a"}, window_index=1
    )[0]
    assert result.track_id == old.track_id
    assert table.tracks[result.track_id].identity == "voice-a"


def test_already_confirmed_global_identity_is_not_split_by_stricter_track_gate():
    table = _table()
    old = table.assign(
        {0: _vector(0)}, {0: 32_000}, {0: 0}, {0: "voice-a"}, window_index=0
    )[0]
    assert old.track_id is not None
    anchor = table.tracks[old.track_id].anchor
    # The caller's authoritative global matcher accepted .68 with sufficient
    # separation. The provisional .70 gate must not undo that positive match.
    for window, samples in ((1, 32_000), (2, 12_000)):
        new = table.assign(
            {0: _similarity(0.68)},
            {0: samples},
            {0: 2},
            {0: "voice-a"},
            window_index=window,
        )[0]
        assert new.track_id == old.track_id
        assert new.reason == "matched_identity"
        track = table.tracks[old.track_id]
        assert track.anchor is anchor
        assert len(track.members) == 1  # Independent .68 long match is retained.


def test_canonical_identity_retains_label_after_provisional_anchor_expiration():
    table = _table()
    old = table.assign(
        {0: _vector(0)}, {0: 32_000}, {0: 0}, {0: "voice-a"}, window_index=0
    )[0]
    new = table.assign(
        {0: _similarity(0.71)}, {0: 12_000}, {0: 2}, {0: "voice-a"}, window_index=7
    )[0]
    assert new.track_id == old.track_id
    assert old.track_id not in table.tracks  # Short evidence cannot seed again.
    later = table.assign(
        {0: _similarity(0.68)}, {0: 32_000}, {0: 1}, {0: "voice-a"}, window_index=8
    )[0]
    assert later.track_id == old.track_id
    assert later.reason == "reestablished_identity"
    assert old.track_id in table.tracks


def test_conflicting_confirmed_history_cannot_reuse_provisional_track():
    table = _table()
    old = table.assign(
        {0: _vector(0)}, {0: 32_000}, {0: 0}, {0: "voice-a"}, window_index=0
    )[0]
    new = table.assign(
        {0: _vector(0)}, {0: 32_000}, {0: 0}, {0: "voice-b"}, window_index=1
    )[0]
    assert new.track_id != old.track_id
    assert new.reason == "identity_conflict"


def test_ambiguous_tracks_remain_unassigned_for_short_speech():
    table = _table()
    table.assign(
        {0: _similarity(0.8, axis=1), 1: _similarity(0.8, axis=2)},
        {0: 32_000, 1: 32_000},
        {0: 0, 1: 1},
        {0: "voice-a", 1: "voice-b"},
        window_index=0,
    )
    result = _assign(table, _vector(0), 1, samples=12_000)
    assert result.track_id is None
    assert result.reason == "match_ambiguous"


def test_table_members_and_identity_numbers_are_bounded_without_reuse():
    table = _table(capacity=1)
    ids = [
        _seed(table, _vector(axis), axis, identity=f"voice-{axis}").track_id
        for axis in range(6)
    ]
    assert len(table.tracks) == 2
    assert len(set(ids)) == 6
    table.clear()
    next_id = _seed(table, _vector(0), 6, identity="voice-new").track_id
    assert next_id not in ids
    for window in range(7, 25):
        assert _assign(table, _vector(0), window).track_id == next_id
    assert len(table.tracks[next_id].members) == 4
    assert len(table.tracks[next_id].weights) == 4


def test_absent_track_expires_and_short_evidence_cannot_recreate_it():
    table = _table()
    old = _seed(table, _vector(0), 0).track_id
    result = _assign(table, _vector(0), 7, samples=12_000)
    assert result.track_id is None
    assert old not in table.tracks


def test_user_stricter_thresholds_are_respected():
    table = SpeakerTrackTable("abc123", 16, 0.98, 0.50)
    old = _seed(table, _vector(0), 0).track_id
    assert _assign(table, _similarity(0.97), 1, samples=12_000).track_id is None
    assert _assign(table, _similarity(0.99), 2, samples=12_000).track_id == old


def test_boundary_cannot_override_a_close_competing_track():
    table = _table()
    results = table.assign(
        {0: _similarity(0.8, axis=1), 1: _similarity(0.8, axis=2)},
        {0: 32_000, 1: 32_000},
        {0: 0, 1: 1},
        {0: "voice-a", 1: "voice-b"},
        window_index=0,
    )
    assert results[0].track_id is not None
    result = table.inherit(
        results[0].track_id,
        island=0,
        channels={0: 0},
        matches={},
        vector=_vector(0),
        identity=None,
        window_index=1,
    )
    assert result.track_id is None
    assert result.reason == "embedding_conflict"


@pytest.mark.parametrize("samples", [0, 7_999, 8_000, 19_199])
def test_no_short_island_can_seed_an_anchor(samples: int):
    table = _table()
    assert _assign(table, _vector(0), 0, samples=samples).track_id is None
    assert not table.tracks


def test_installing_global_reference_then_assigning_it_does_not_count_it_twice():
    table = _table()
    anchor, member = _vector(0), _similarity(0.68)
    table.install_identity("voice-a", anchor, (member,), window_index=0)
    result = table.assign(
        {0: member}, {0: 32000}, {0: 0}, {0: "voice-a"}, window_index=0
    )[0]
    track = table.tracks[result.track_id]
    assert len(track.members) == 1
    # A query resembling only this one unusual reference must remain unknown.
    query = _similarity(0.4)
    assert np.dot(query, member) > 0.9
    assert _assign(table, query, 1).track_id is None
