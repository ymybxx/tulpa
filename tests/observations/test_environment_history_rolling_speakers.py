from types import SimpleNamespace

import numpy as np
import pytest

from fish_memory.observations.environment_history_rolling_speakers import (
    RollingSpeakerDiarizer,
)
from fish_memory.observations.environment_history_speakers import (
    SpeakerDiarizationError,
    SpeakerWindowOrderError,
)


class Kernels:
    def __init__(self, *, overlap=False):
        self.extract_calls = []
        self.infer_calls = 0
        self.overlap = overlap
        self.fail = False

    def _infer(self, samples):
        self.infer_calls += 1
        indexes = np.minimum(
            (495.5 + np.arange(589) * 270).astype(int), len(samples) - 1
        )
        labels = np.where(abs(samples[indexes]) > 0.01, 4 if self.overlap else 1, 0)
        scores = np.full((589, 7), -20, dtype=np.float32)
        scores[np.arange(589), labels] = 0
        return scores

    def _extract(self, samples):
        if self.fail:
            raise SpeakerDiarizationError("test extraction failure")
        self.extract_calls.append(samples.copy())
        return np.array([1, 0] if samples.sum() >= 0 else [0, 1], dtype=np.float32)

    def close(self):
        pass


def make_diarizer(**kwargs):
    diarizer = RollingSpeakerDiarizer(SimpleNamespace())
    diarizer._kernels = Kernels(**kwargs)
    return diarizer


def pcm(*regions):
    values = np.zeros(160000, dtype="<i2")
    for start, end, amplitude in regions:
        values[int(start * 16000) : int(end * 16000)] = amplitude
    return values.tobytes()


def test_short_utterance_is_not_dropped_and_silence_delivers_pending_revision():
    diarizer = make_diarizer()
    first = diarizer.segment(pcm((8, 8.5, 1000)), stream_id="a", window_index=0)
    assert first and all(s.speaker_id is None for s in first)
    diarizer.observe_silence(stream_id="a", window_index=1)
    revision = next(r for r in diarizer.window_revisions if r.window_index == 0)
    assert all(s.speaker_id is None for s in revision.spans)
    assert len(diarizer._state.evidence) == 1
    assert len(diarizer._kernels.extract_calls[0]) == 24000
    item = next(iter(diarizer._state.evidence.values()))
    assert item.end - item.start < 16000  # Repeated padding is not extra evidence.
    diarizer.segment(pcm((0, 3, 1000)), stream_id="a", window_index=2)
    revised = next(r for r in diarizer.window_revisions if r.window_index == 0)
    assert any(s.speaker_id for s in revised.spans)
    assert item.end < diarizer._state.pcm_start  # Resolved from vectors, not old PCM.


def test_retry_and_failed_extraction_cannot_mutate_committed_identity_state():
    diarizer = make_diarizer()
    data = pcm((0, 4, 1000))
    first = diarizer.segment(data, stream_id="a", window_index=0)
    before = diarizer._state
    calls = len(diarizer._kernels.extract_calls)
    assert diarizer.segment(data, stream_id="a", window_index=0) is first
    assert len(diarizer._kernels.extract_calls) == calls
    with pytest.raises(SpeakerWindowOrderError):
        diarizer.segment(pcm((1, 4, -1000)), stream_id="a", window_index=0)
    diarizer._kernels.fail = True
    with pytest.raises(SpeakerDiarizationError):
        diarizer.segment(pcm((0, 4, -1000)), stream_id="a", window_index=1)
    assert diarizer._state is before
    diarizer._kernels.fail = False
    diarizer.segment(pcm((0, 4, -1000)), stream_id="a", window_index=1)
    assert diarizer._state.last_index == 1


def test_overlap_is_not_a_single_person_and_does_not_train_voice_profiles():
    diarizer = make_diarizer(overlap=True)
    spans = diarizer.segment(pcm((0, 4, 1000)), stream_id="a", window_index=0)
    assert any(s.overlap for s in spans)
    assert all(s.speaker_id is None and s.track_id is None for s in spans)
    assert not diarizer._state.evidence
    assert not diarizer._state.bank.profiles


def test_long_stream_keeps_pcm_and_embeddings_bounded_and_stream_reset_isolated():
    diarizer = make_diarizer()
    data = pcm((0, 10, 1000))
    for index in range(40):
        diarizer.segment(data, stream_id="a", window_index=index)
        assert len(diarizer._state.pcm) <= 20 * 16000
        assert len(diarizer._state.evidence) <= 512
        assert len(diarizer._state.windows) <= 31
    old = {s.speaker_id for s in diarizer._state.last_spans if s.speaker_id}
    new = {
        s.speaker_id
        for s in diarizer.segment(data, stream_id="b", window_index=0)
        if s.speaker_id
    }
    assert old and new and not old & new
    assert not diarizer.window_revisions


def test_silence_and_cache_expiry_do_not_recluster_old_identity_evidence(monkeypatch):
    from fish_memory.observations.environment_history_voice_clusters import (
        VoiceClusterBank,
    )

    diarizer = make_diarizer()
    diarizer.segment(pcm((0, 3, 1000)), stream_id="a", window_index=0)
    diarizer.observe_silence(stream_id="a", window_index=1)
    before = diarizer._state.published[0]

    def unexpected_recluster(*args):
        raise AssertionError("expiry is not new acoustic evidence")

    monkeypatch.setattr(VoiceClusterBank, "reconcile", unexpected_recluster)
    for index in range(2, 33):
        diarizer.observe_silence(stream_id="a", window_index=index)
        if 0 in diarizer._state.published:
            assert diarizer._state.published[0] == before
    assert not diarizer._state.evidence


def test_same_capture_resuming_after_long_gap_creates_new_acoustic_evidence():
    diarizer = make_diarizer()
    initial = diarizer.segment(pcm((0, 4, 1000)), stream_id="a", window_index=0)
    later = diarizer.segment(pcm((0, 4, -1000)), stream_id="a", window_index=10)
    assert any(s.speaker_id for s in later)
    assert {s.speaker_id for s in initial if s.speaker_id}.isdisjoint(
        s.speaker_id for s in later if s.speaker_id
    )
