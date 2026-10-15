from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from fish_memory.config import Settings
from fish_memory.observations import environment_history_speakers as speakers


def _vector(index: int) -> np.ndarray:
    value = np.zeros(192, dtype=np.float32)
    value[index] = 1
    return value


def _scores(*turns: tuple[float, float, int]) -> np.ndarray:
    result = np.full((speakers._FRAME_COUNT, 7), -12, dtype=np.float32)
    result[:, 0] = 0
    centers = (
        np.arange(speakers._FRAME_COUNT) * speakers._FRAME_SHIFT
        + speakers._FRAME_SIZE / 2
    ) / speakers.SAMPLE_RATE
    for start, end, label in turns:
        mask = (centers >= start) & (centers < end)
        result[mask, :] = -12
        result[mask, label] = 0
    return result


def _uncertain_scores(*turns: tuple[float, float, int]) -> np.ndarray:
    """Keep the speech/channel but remove a substantial reliable center region."""
    result = _scores(*turns)
    centers = (
        np.arange(speakers._FRAME_COUNT) * speakers._FRAME_SHIFT
        + speakers._FRAME_SIZE / 2
    ) / speakers.SAMPLE_RATE
    for start, end, label in turns:
        middle, radius = (start + end) / 2, (end - start) * 0.1
        probabilities = np.full(7, 0.075)
        probabilities[label] = 0.55
        result[(centers > middle - radius) & (centers < middle + radius)] = np.log(
            probabilities
        )
    return result


def _pcm(seconds: float = 10) -> bytes:
    samples = np.sin(np.arange(int(seconds * 16_000)) * 0.08) * 8_000
    return samples.astype("<i2").tobytes()


class _FakeEncoder:
    def __init__(self) -> None:
        self.vectors: list[np.ndarray] = []
        self.lengths: list[int] = []

    def extract(self, samples: np.ndarray) -> np.ndarray:
        self.lengths.append(len(samples))
        return self.vectors.pop(0)


@pytest.fixture
def raw_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    settings = cast(Settings, SimpleNamespace(owner_voice_dir=str(tmp_path / "owner")))
    diarizer = speakers.LocalSpeakerDiarizer(settings)
    encoder = _FakeEncoder()
    diarizer._encoder = encoder
    state = SimpleNamespace(
        diarizer=diarizer, encoder=encoder, scores=_scores(), calls=0
    )

    def infer(samples: np.ndarray) -> np.ndarray:
        state.calls += 1
        return state.scores

    monkeypatch.setattr(diarizer, "_infer", infer)
    return state


@pytest.fixture
def engine(raw_engine: Any) -> Any:
    return raw_engine


def test_local_speaker_permutations_keep_global_ids(engine: Any) -> None:
    engine.scores = _scores((0, 4, 1), (5, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    first = engine.diarizer.segment(_pcm(), stream_id="movie", window_index=0)
    assert len(first) == 2
    first_a, first_b = first[0].speaker_id, first[1].speaker_id
    assert first_a is not None and first_b is not None and first_a != first_b
    assert first_a.startswith("声音") and first_a.endswith("-01")

    # Same people, opposite model-local output channels and speaking order.
    engine.scores = _scores((0, 4, 1), (5, 10, 2))
    engine.encoder.vectors = [_vector(1), _vector(0)]
    second = engine.diarizer.segment(_pcm(), stream_id="movie", window_index=1)
    assert [span.speaker_id for span in second] == [first_b, first_a]
    assert len(engine.diarizer._voices) == 2
    assert len(engine.encoder.lengths) == 4  # Embeddings run even on one chunk.


def test_one_speaker_still_extracts_embedding_each_window(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert first[0].speaker_id == second[0].speaker_id
    assert len(engine.encoder.lengths) == 2
    assert max(engine.encoder.lengths) == 5 * 16_000


def test_repeat_is_cached_and_out_of_order_does_not_change_state(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=3)
    assert engine.diarizer.segment(_pcm(), stream_id="a", window_index=3) is first
    assert engine.calls == 1
    namespace = engine.diarizer._namespace
    with pytest.raises(speakers.SpeakerWindowOrderError):
        engine.diarizer.segment(_pcm(9), stream_id="a", window_index=3)
    with pytest.raises(speakers.SpeakerWindowOrderError):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)
    assert engine.diarizer._namespace == namespace
    assert engine.diarizer._last_index == 3
    assert len(engine.diarizer._voices) == 1
    assert engine.calls == 1


def test_gaps_stream_changes_and_reset_do_not_recycle_ids(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)] * 5
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    gap = engine.diarizer.segment(_pcm(), stream_id="a", window_index=7)
    assert gap[0].speaker_id == first[0].speaker_id  # Six skipped windows.
    long_gap = engine.diarizer.segment(_pcm(), stream_id="a", window_index=15)
    assert long_gap[0].speaker_id != gap[0].speaker_id
    switched = engine.diarizer.segment(_pcm(), stream_id="b", window_index=0)
    assert switched[0].speaker_id != long_gap[0].speaker_id
    engine.diarizer.reset()
    reset = engine.diarizer.segment(_pcm(), stream_id="b", window_index=1)
    assert reset[0].speaker_id != switched[0].speaker_id


def test_short_and_low_confidence_turns_remain_unknown(engine: Any) -> None:
    engine.scores = _scores((0, 0.8, 1), (1, 1.7, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 2  # Distinct local people retain distinct ASR requests.
    assert all(span.speaker_id is None for span in spans)
    assert spans[0].local_speaker_id != spans[1].local_speaker_id
    assert all(span.reason == "match_below_threshold" for span in spans)
    assert len(engine.encoder.lengths) == 2  # Retained for future confirmation.
    engine.scores = np.zeros((speakers._FRAME_COUNT, 7), dtype=np.float32)
    unknown = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(unknown) == 1
    assert unknown[0].speaker_id is None
    assert len(engine.encoder.lengths) == 2


def test_overlap_is_unknown_and_excluded_from_voice_embeddings(engine: Any) -> None:
    engine.scores = _scores((0, 3, 1), (3, 6, 4), (6, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 3
    assert spans[0].speaker_id is not None
    assert spans[1].overlap and spans[1].speaker_id is None
    assert spans[2].speaker_id is not None
    assert spans[0].speaker_id != spans[2].speaker_id
    assert all(
        left.end_sample <= right.start_sample
        for left, right in zip(spans, spans[1:], strict=False)
    )
    assert all(length < 4 * 16_000 for length in engine.encoder.lengths)


def test_confidence_dips_do_not_fragment_asr_turns(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    # A brief dip on each second leaves the argmax speaker intact; these
    # uncertain frames must not create separate 17 ms recognition requests.
    for index in range(80, speakers._FRAME_COUNT, 60):
        engine.scores[index, :] = np.log(
            np.array([0.4, 0.55, 0.01, 0.01, 0.01, 0.01, 0.01])
        )
    engine.encoder.vectors = [_vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 1
    assert spans[0].speaker_id is None  # Fragmented clean evidence awaits confirmation.
    assert spans[0].start_sample == 0 and spans[0].end_sample == 160_000
    assert len(engine.encoder.lengths) == 1


def test_short_pause_joins_same_voice_but_preserves_real_speaker_change(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 2, 1), (2.15, 4, 1), (4.1, 7, 2), (7.1, 10, 1))
    engine.encoder.vectors = [_vector(0), _vector(1), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 3
    assert spans[0].speaker_id == spans[2].speaker_id
    assert spans[0].speaker_id != spans[1].speaker_id


def test_same_local_channel_after_long_pause_can_be_a_different_person(
    engine: Any,
) -> None:
    # The segmentation model missed the change. Separate island embeddings
    # must still distinguish these people rather than mix their PCM together.
    engine.scores = _scores((0, 3, 1), (6, 10, 1))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 2
    assert spans[0].speaker_id is not None
    assert spans[1].speaker_id is not None
    assert spans[0].speaker_id != spans[1].speaker_id
    assert len(engine.encoder.lengths) == 2
    assert spans[0].track_id is not None
    assert spans[0].track_id != spans[1].track_id


def test_short_tail_never_inherits_later_person_on_same_local_channel(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 0.8, 1), (4, 10, 1))
    engine.encoder.vectors = [_vector(1), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 2
    assert spans[0].speaker_id is None
    assert spans[1].speaker_id is not None
    assert len(engine.encoder.lengths) == 2


def test_first_heard_voice_matches_across_separately_verified_islands(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 3, 1), (6, 10, 1))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert len(spans) == 2
    assert spans[0].speaker_id is not None
    assert spans[0].speaker_id == spans[1].speaker_id
    assert len(engine.encoder.lengths) == 2
    assert len(engine.diarizer._voices) == 1
    assert spans[0].track_id == spans[1].track_id


def test_similar_local_people_cannot_share_one_global_id(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.scores = _scores((0, 4, 1), (5, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert all(span.speaker_id is None for span in spans)
    assert len(engine.diarizer._voices) == 1


def test_ambiguous_template_margin_does_not_update_or_create_identity(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 4, 1), (5, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [(_vector(0) + _vector(1)) / np.sqrt(2)]
    ambiguous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert ambiguous[0].speaker_id is None
    assert len(engine.diarizer._voices) == 2
    assert all(voice.updates == 1 for voice in engine.diarizer._voices.values())


def test_capacity_is_bounded_without_recycling_historical_ids(engine: Any) -> None:
    engine.diarizer._capacity = 2
    engine.scores = _scores((0, 10, 1))
    ids = []
    for index in range(5):
        engine.encoder.vectors = [_vector(index)]
        spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=index)
        ids.append(spans[0].speaker_id)
    assert all(identity is not None for identity in ids)
    assert len(set(ids)) == 5
    assert len(engine.diarizer._voices) == 2
    assert set(engine.diarizer._voices) == set(ids[-2:])


def test_capacity_protects_ids_assigned_in_the_current_window(engine: Any) -> None:
    engine.diarizer._capacity = 1
    engine.scores = _scores((0, 4, 1), (5, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert spans[0].speaker_id is not None and spans[1].speaker_id is None
    assert list(engine.diarizer._voices) == [spans[0].speaker_id]


def test_recognized_voice_is_refreshed_before_lru_eviction(engine: Any) -> None:
    engine.diarizer._capacity = 2
    engine.scores = _scores((0, 10, 1))
    ids = []
    for index, voice in enumerate((0, 1, 0, 2)):
        engine.encoder.vectors = [_vector(voice)]
        span = engine.diarizer.segment(_pcm(), stream_id="a", window_index=index)[0]
        ids.append(span.speaker_id)
    assert ids[0] == ids[2]
    assert ids[0] in engine.diarizer._voices
    assert ids[1] not in engine.diarizer._voices
    assert ids[3] in engine.diarizer._voices


def test_nonfinite_inference_or_embeddings_never_enter_templates(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    bad = _vector(1)
    bad[4] = np.nan
    engine.encoder.vectors = [bad]
    with pytest.raises(speakers.SpeakerDiarizationError):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(engine.diarizer._voices) == 1
    assert engine.diarizer._last_index == 0
    engine.scores[3, 2] = np.inf
    with pytest.raises(speakers.SpeakerDiarizationError):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(engine.diarizer._voices) == 1
    assert engine.diarizer._last_index == 0


def test_silence_and_empty_audio_do_not_load_models(engine: Any) -> None:
    assert engine.diarizer.segment(bytes(320_000), stream_id="a", window_index=0) == ()
    assert engine.diarizer.segment(b"", stream_id="a", window_index=1) == ()
    assert not engine.calls
    assert not engine.encoder.lengths
    assert not engine.diarizer._voices


@pytest.mark.parametrize(
    "pcm",
    [b"x", bytes(320_002), "not pcm"],
    ids=["unaligned", "oversized", "wrong-type"],
)
def test_invalid_pcm_rejected_before_model_load(engine: Any, pcm: Any) -> None:
    with pytest.raises(speakers.SpeakerDiarizationError):
        engine.diarizer.segment(pcm, stream_id="a", window_index=0)
    assert not engine.calls


def test_partial_window_has_python_integer_offsets_inside_real_pcm(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    spans = engine.diarizer.segment(_pcm(2), stream_id="a", window_index=0)
    assert spans[0].start_sample == 0
    assert spans[-1].end_sample == 32_000
    assert all(
        type(span.start_sample) is int and type(span.end_sample) is int
        for span in spans
    )


def test_close_releases_sessions_and_all_voice_state(engine: Any) -> None:
    engine.scores = _scores((0, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.diarizer._session = object()
    engine.diarizer.close()
    assert engine.diarizer._session is None
    assert engine.diarizer._encoder is None
    assert not engine.diarizer._voices
    assert engine.diarizer._last_index is None


def test_encoder_reuses_model_file_without_touching_owner_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    model = owner / speakers.EMBEDDING_MODEL_ID
    model.write_bytes(b"public model")
    (owner / "owner-voice.json").write_text("not valid JSON: must never be read")
    monkeypatch.setattr(speakers, "EMBEDDING_MODEL_SIZE", model.stat().st_size)
    monkeypatch.setattr(
        speakers,
        "EMBEDDING_MODEL_SHA256",
        hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    assert speakers._prepare_embedding(tmp_path / "speakers", owner) == model


def test_verified_model_rejects_invalid_size_and_hash(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert speakers._verified(target, 3, digest)
    assert not speakers._verified(target, 4, digest)
    assert not speakers._verified(target, 3, "0" * 64)


def test_verified_model_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    assert not speakers._verified(link, 3, digest)


def test_prepare_extracts_only_pinned_model_and_preserves_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:bz2") as archive:
        for name, value in (
            (f"{speakers._ARCHIVE_ROOT}/model.onnx", b"model"),
            (f"{speakers._ARCHIVE_ROOT}/LICENSE", b"MIT license"),
            ("../../outside", b"must never extract"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    monkeypatch.setattr(speakers, "_download", lambda *args: body.getvalue())
    monkeypatch.setattr(speakers, "MODEL_SIZE", 5)
    monkeypatch.setattr(speakers, "MODEL_SHA256", hashlib.sha256(b"model").hexdigest())
    model = speakers._prepare_segmentation(tmp_path / "models")
    assert model.read_bytes() == b"model"
    assert (model.parent / "segmentation-3.0.LICENSE").read_text() == "MIT license"
    assert sorted(path.name for path in model.parent.iterdir()) == [
        speakers.MODEL_ID,
        "segmentation-3.0.LICENSE",
    ]


def test_download_bounds_size_and_verifies_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(speakers, "urlopen", lambda *args, **kwargs: io.BytesIO(b"abc"))
    digest = hashlib.sha256(b"abc").hexdigest()
    assert speakers._download("https://example.invalid/model", 3, digest) == b"abc"
    with pytest.raises(speakers.SpeakerDiarizationError):
        speakers._download("https://example.invalid/model", 2, digest)
    with pytest.raises(speakers.SpeakerDiarizationError):
        speakers._download("https://example.invalid/model", 3, "0" * 64)


def test_direct_onnx_inference_pads_once_and_validates_shape(engine: Any) -> None:
    settings = cast(Settings, SimpleNamespace(owner_voice_dir="unused"))
    diarizer = speakers.LocalSpeakerDiarizer(settings)
    fed: list[np.ndarray] = []

    class Session:
        def run(
            self, names: list[str], inputs: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            assert names == ["y"]
            fed.append(inputs["x"])
            return [_scores((0, 10, 1))[np.newaxis, :, :]]

    diarizer._session = Session()
    diarizer._encoder = object()
    values = np.ones(32_000, dtype=np.float32)
    output = diarizer._infer(values)
    assert output.shape == (speakers._FRAME_COUNT, 7)
    assert fed[0].shape == (1, 1, 160_000)
    assert np.all(fed[0][0, 0, :32_000] == 1)
    assert np.all(fed[0][0, 0, 32_000:] == 0)


def test_known_voice_short_utterance_matches_without_updating_template(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    identity = first.speaker_id
    original = engine.diarizer._voices[identity]
    engine.scores = _scores((1, 1.8, 2))
    engine.encoder.vectors = [_vector(0)]
    short = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert short.speaker_id == identity
    assert short.reason == "matched"
    assert short.identity_source == "embedding"
    assert 0.5 <= short.clean_seconds < 1.2
    assert short.best_similarity == pytest.approx(1)
    assert short.second_similarity is None
    assert engine.diarizer._voices[identity] is original
    assert original.updates == 1
    assert len(engine.encoder.lengths) == 2


def test_first_short_utterance_matches_later_long_voice_only_with_own_embedding(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 0.8, 1), (4, 10, 1))
    # Long evidence is processed first, then the independently observed short.
    engine.encoder.vectors = [_vector(0), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert spans[0].speaker_id is not None
    assert spans[0].speaker_id == spans[1].speaker_id
    assert spans[0].reason == "matched" and spans[1].reason == "quality_anchor"
    assert len(engine.diarizer._voices) == 1
    assert engine.diarizer._voices[spans[0].speaker_id].updates == 1
    assert engine.encoder.lengths[0] > engine.encoder.lengths[1]


def test_short_utterance_cannot_steal_long_local_person_identity(engine: Any) -> None:
    engine.scores = _scores((0, 0.8, 2), (4, 10, 1))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert spans[0].speaker_id is None
    assert spans[0].reason == "local_conflict"
    assert spans[0].best_similarity == pytest.approx(1)
    assert spans[1].speaker_id is not None
    assert engine.diarizer._voices[spans[1].speaker_id].updates == 1


def test_short_competing_local_people_remain_separate_and_unassigned(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.scores = _scores((1, 1.8, 1), (2, 2.8, 2))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(spans) == 2
    assert spans[0].local_speaker_id != spans[1].local_speaker_id
    assert all(
        span.speaker_id is None and span.reason == "local_conflict" for span in spans
    )
    assert all(voice.updates == 1 for voice in engine.diarizer._voices.values())


@pytest.mark.parametrize("score", [0.54, 0.60, 0.68])
def test_short_weak_match_does_not_create_or_update_voice(
    engine: Any, score: float
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    identity = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[
        0
    ].speaker_id
    engine.scores = _scores((1, 1.8, 1))
    engine.encoder.vectors = [_vector(0) * score + _vector(1) * np.sqrt(1 - score**2)]
    short = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert short.speaker_id is None
    assert short.reason == "match_below_threshold"
    assert short.best_similarity == pytest.approx(score)
    assert list(engine.diarizer._voices) == [identity]
    assert engine.diarizer._voices[identity].updates == 1


def test_long_gray_zone_does_not_create_identity_and_preserves_diagnostics(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    identity = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[
        0
    ].speaker_id
    engine.encoder.vectors = [_vector(0) * 0.6 + _vector(1) * 0.8]
    uncertain = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert uncertain.speaker_id is None
    assert uncertain.reason == "match_below_threshold"
    assert uncertain.best_similarity == pytest.approx(0.6)
    assert uncertain.clean_seconds >= 1.2
    assert list(engine.diarizer._voices) == [identity]


def test_unknown_model_channels_and_islands_keep_scoped_local_ids(engine: Any) -> None:
    engine.scores = _scores((0, 0.4, 1), (0.5, 0.9, 2), (2, 2.4, 1))
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(first) == 3
    assert len({span.local_speaker_id for span in first}) == 3
    assert {span.local_speaker_id for span in first}.isdisjoint(
        span.local_speaker_id for span in second
    )
    assert all(span.speaker_id is None for span in first + second)


def test_failed_short_embedding_rolls_back_long_updates_new_identities_and_lru(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    original_table = engine.diarizer._voices.copy()
    original_next = engine.diarizer._next_identity
    original_namespace = engine.diarizer._namespace
    # Matching the long turn would update its template before short extraction;
    # a subsequent invalid embedding must roll back that entire transaction.
    engine.scores = _scores((0, 0.8, 1), (2, 5, 1), (6, 10, 2))
    engine.encoder.vectors = [_vector(0), _vector(1), np.full(192, np.nan)]
    with pytest.raises(speakers.SpeakerDiarizationError):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert engine.diarizer._namespace == original_namespace
    assert engine.diarizer._next_identity == original_next
    assert list(engine.diarizer._voices) == list(original_table)
    assert all(
        engine.diarizer._voices[key] is voice for key, voice in original_table.items()
    )
    assert engine.diarizer._last_index == 0
    assert engine.diarizer._last_spans is first
    engine.encoder.vectors = [_vector(0), _vector(1), _vector(0)]
    retry = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert retry[0].speaker_id == first[0].speaker_id
    assert retry[2].speaker_id.endswith(f"-{original_next:02d}")


def test_failed_stream_change_preserves_previous_successful_voice_state(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    original_namespace = engine.diarizer._namespace
    engine.encoder.vectors = [np.full(192, np.inf)]
    with pytest.raises(speakers.SpeakerDiarizationError):
        engine.diarizer.segment(_pcm(), stream_id="b", window_index=0)
    assert engine.diarizer._stream_id == "a"
    assert engine.diarizer._namespace == original_namespace
    assert engine.diarizer._last_spans is first
    assert list(engine.diarizer._voices) == [first[0].speaker_id]


def test_capacity_reason_keeps_clean_evidence_and_unknown_local_turn(
    engine: Any,
) -> None:
    engine.diarizer._capacity = 1
    engine.scores = _scores((1, 4, 1), (5, 9, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    first, second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert first.reason == "quality_anchor"
    assert second.reason == "capacity" and second.speaker_id is None
    assert second.clean_seconds >= 1.2
    assert second.local_speaker_id is not None
    assert second.identity_source == "local"


def _set_window_and_boundary_scores(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
    current: np.ndarray,
    boundary: np.ndarray,
) -> None:
    outputs = iter((current, boundary))

    def infer(samples: np.ndarray) -> np.ndarray:
        engine.calls += 1
        return next(outputs)

    monkeypatch.setattr(engine.diarizer, "_infer", infer)


def test_continuous_boundary_recovers_short_tail_without_embedding_or_template_update(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    voice = engine.diarizer._voices[previous.speaker_id]
    _set_window_and_boundary_scores(
        engine, monkeypatch, _scores((0, 0.4, 2)), _scores((0, 2.4, 3))
    )
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert current.speaker_id == previous.speaker_id
    assert current.identity_source == "boundary"
    assert current.reason == "boundary_continuity"
    assert current.boundary_reason == "continuous"
    assert current.clean_seconds < 0.5
    assert current.start_sample == 0 and current.end_sample < 16_000
    assert len(engine.encoder.lengths) == 1
    assert engine.diarizer._voices[previous.speaker_id] is voice
    assert engine.calls == 3


def test_boundary_does_not_override_conflicting_short_voice_embedding(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    _set_window_and_boundary_scores(
        engine, monkeypatch, _scores((0, 0.8, 2)), _scores((0, 2.8, 3))
    )
    engine.encoder.vectors = [_vector(1)]
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert current.speaker_id is None
    assert current.reason == "match_below_threshold"
    assert current.boundary_reason == "embedding_conflict"
    assert current.best_similarity == pytest.approx(0)
    assert list(engine.diarizer._voices) == [previous.speaker_id]


def test_boundary_respects_long_current_local_cannot_link(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    _set_window_and_boundary_scores(
        engine, monkeypatch, _scores((0, 0.4, 2), (4, 8, 1)), _scores((0, 2.4, 3))
    )
    engine.encoder.vectors = [_vector(0)]
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert spans[0].speaker_id is None
    assert spans[0].reason == "local_conflict"
    assert spans[0].boundary_reason == "local_conflict"
    assert spans[1].speaker_id == previous.speaker_id


def test_clear_boundary_and_rejected_order_keep_templates_but_discard_pcm(
    engine: Any,
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)
    assert engine.diarizer._boundary._pcm
    engine.diarizer.clear_boundary()
    assert not engine.diarizer._boundary._pcm
    assert list(engine.diarizer._voices) == [first[0].speaker_id]
    engine.encoder.vectors = [_vector(0)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=3)
    assert engine.diarizer._boundary._pcm
    with pytest.raises(speakers.SpeakerWindowOrderError):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)
    assert not engine.diarizer._boundary._pcm
    assert list(engine.diarizer._voices) == [first[0].speaker_id]
    engine.scores = _scores((0, 0.4, 1))
    short = engine.diarizer.segment(_pcm(), stream_id="a", window_index=4)[0]
    assert short.speaker_id is None


def test_failed_boundary_inference_keeps_current_segmentation_and_commit(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    counter = 0

    def infer(samples: np.ndarray) -> np.ndarray:
        nonlocal counter
        counter += 1
        if counter == 1:
            return _scores((0, 0.4, 2))
        raise speakers.SpeakerDiarizationError("boundary unavailable")

    monkeypatch.setattr(engine.diarizer, "_infer", infer)
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert current.speaker_id is None and current.reason == "short_evidence"
    assert current.boundary_reason == "inference_failed"
    assert current.local_speaker_id is not None
    assert engine.diarizer._last_index == 1
    assert list(engine.diarizer._voices) == [previous.speaker_id]


def test_short_match_caps_never_relax_custom_strict_thresholds(engine: Any) -> None:
    engine.diarizer._threshold = 0.98
    engine.diarizer._margin = 0.50
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    engine.scores = _scores((1, 1.8, 1))
    engine.encoder.vectors = [_vector(0) * 0.96 + _vector(1) * 0.28]
    short = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert short.speaker_id is None
    assert short.reason == "match_below_threshold"
    assert short.best_similarity == pytest.approx(0.96)
    assert list(engine.diarizer._voices) == [previous.speaker_id]
    engine.diarizer._voices["second"] = speakers._Voice(
        _vector(0) * 0.45 + _vector(2) * np.sqrt(1 - 0.45**2),
        _vector(0) * 0.45 + _vector(2) * np.sqrt(1 - 0.45**2),
    )
    # Similarity gap 1-.45=.55 meets the configured .50; .45 would not.
    engine.encoder.vectors = [_vector(0)]
    matched = engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)[0]
    assert matched.speaker_id == previous.speaker_id
    engine.diarizer._voices["second"] = speakers._Voice(
        _vector(0) * 0.55 + _vector(2) * np.sqrt(1 - 0.55**2),
        _vector(0) * 0.55 + _vector(2) * np.sqrt(1 - 0.55**2),
    )
    engine.encoder.vectors = [_vector(0)]
    ambiguous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=3)[0]
    assert ambiguous.speaker_id is None
    assert ambiguous.reason == "match_ambiguous"


@pytest.mark.parametrize(
    ("old_similarity", "other_similarity", "expected_boundary"),
    [(0.66, 0.80, False), (0.80, 0.75, False), (0.80, 0.66, True)],
)
def test_boundary_requires_old_identity_to_win_ordinary_cam_margin(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
    old_similarity: float,
    other_similarity: float,
    expected_boundary: bool,
) -> None:
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    original = engine.diarizer._voices[previous.speaker_id]
    query = _vector(0) * old_similarity + _vector(1) * np.sqrt(1 - old_similarity**2)
    alternative = query * other_similarity + _vector(2) * np.sqrt(
        1 - other_similarity**2
    )
    engine.diarizer._voices["other"] = speakers._Voice(alternative, alternative)
    _set_window_and_boundary_scores(
        engine, monkeypatch, _scores((0, 0.8, 2)), _scores((0, 2.8, 3))
    )
    engine.encoder.vectors = [query]
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    if expected_boundary:
        assert current.speaker_id == previous.speaker_id
        assert current.reason == "boundary_continuity"
        assert current.identity_source == "boundary"
        assert current.boundary_reason == "continuous"
    else:
        assert current.speaker_id is None
        assert current.reason == "match_ambiguous"
        assert current.boundary_reason == "embedding_conflict"
    assert current.best_similarity == pytest.approx(
        max(old_similarity, other_similarity)
    )
    assert current.second_similarity == pytest.approx(
        min(old_similarity, other_similarity)
    )
    assert engine.diarizer._voices[previous.speaker_id] is original
    assert original.updates == 1
    assert len(engine.diarizer._voices) == 2


def test_short_match_candidate_accepts_point_71_without_changing_voice_template(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    original = engine.diarizer._voices[previous.speaker_id]
    engine.scores = _scores((1, 1.8, 1))
    engine.encoder.vectors = [_vector(0) * 0.71 + _vector(1) * np.sqrt(1 - 0.71**2)]
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert current.speaker_id == previous.speaker_id
    assert current.reason == "matched"
    assert current.best_similarity == pytest.approx(0.71)
    assert engine.diarizer._voices[previous.speaker_id] is original


def test_boundary_guard_exhausted_short_turn_is_not_low_segmentation_confidence(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 0.08, 1))
    current = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    assert current.speaker_id is None
    assert current.reason == "short_evidence"
    assert current.clean_seconds == 0
    assert current.local_speaker_id is not None
    assert not engine.encoder.lengths


def _seed_competing_voices(engine: Any) -> np.ndarray:
    engine.scores = _scores((1, 4, 1), (5, 9, 2))
    engine.encoder.vectors = [_vector(0), _vector(1)]
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    return (_vector(0) + _vector(1)) / np.sqrt(2)


def test_global_ambiguity_does_not_publish_another_person_for_natural_pauses(
    engine: Any,
) -> None:
    ambiguous = _seed_competing_voices(engine)
    engine.scores = _scores((0, 0.8, 1), (1.5, 5.2, 1), (6, 6.8, 1), (8, 8.8, 1))
    engine.encoder.vectors = [ambiguous] * 4
    turns = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert len(turns) == 4
    assert all(turn.speaker_id is None and turn.track_id is None for turn in turns)
    assert all(turn.reason == "match_ambiguous" for turn in turns)
    assert len({turn.local_speaker_id for turn in turns}) == 4
    assert len(engine.diarizer._voices) == 2


def test_pending_voice_is_not_a_boundary_identity(raw_engine: Any) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    old = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    engine.scores = _scores((0, 0.4, 2))
    new = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert old.speaker_id is None and old.track_id is None
    assert new.speaker_id is None and new.track_id is None
    assert new.boundary_reason == "no_boundary_candidate"
    assert len(engine.encoder.lengths) == 1


def test_gap_reestablishes_known_voice_with_canonical_label_and_new_anchor(
    engine: Any,
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 2
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    old_anchor = engine.diarizer._tracks.tracks[first.track_id]
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)[0]
    assert first.speaker_id == second.speaker_id
    assert first.track_id == second.track_id
    assert engine.diarizer._tracks.tracks[second.track_id] is not old_anchor
    assert second.track_reason == "matched_identity"


def test_gap_discards_unrecognized_track_instead_of_stitching_it_across_missing_audio(
    engine: Any,
) -> None:
    ambiguous = _seed_competing_voices(engine)
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [ambiguous] * 2
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=3)[0]
    assert first.speaker_id is None and second.speaker_id is None
    assert first.track_id is None and second.track_id is None
    assert not engine.diarizer.revisions


def test_overlap_has_no_track_and_independent_local_people_do_not_share_track(
    engine: Any,
) -> None:
    engine.scores = _scores((0, 3, 1), (3, 5, 4), (5, 9, 2))
    engine.encoder.vectors = [_vector(0)] * 2
    first, overlap, second = engine.diarizer.segment(
        _pcm(), stream_id="a", window_index=0
    )
    assert overlap.track_id is None
    assert first.track_id and second.track_id and first.track_id != second.track_id


def test_track_transaction_rolls_back_after_post_assignment_failure(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    table = engine.diarizer._tracks
    snapshot = table.snapshot()
    original = engine.diarizer._apply_boundary

    def fail(*args: object) -> None:
        raise RuntimeError("after track update")

    monkeypatch.setattr(engine.diarizer, "_apply_boundary", fail)
    engine.encoder.vectors = [_vector(0)]
    with pytest.raises(RuntimeError, match="after track update"):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert engine.diarizer._tracks is table
    assert table.next_id == snapshot[2]
    assert table.canonical == snapshot[1]
    assert table.tracks.keys() == snapshot[0].keys()
    assert all(table.tracks[key] is track for key, track in snapshot[0].items())
    assert engine.diarizer._last_spans is first
    monkeypatch.setattr(engine.diarizer, "_apply_boundary", original)
    engine.encoder.vectors = [_vector(0)]
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert second[0].track_id == first[0].track_id


def test_evicted_global_template_does_not_erase_independently_verified_track_seam(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.diarizer._capacity = 1
    engine.scores = _scores((7, 10, 1))
    engine.encoder.vectors = [_vector(0)]
    previous = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    _set_window_and_boundary_scores(
        engine, monkeypatch, _scores((0, 0.4, 2), (4, 8, 1)), _scores((0, 2.4, 3))
    )
    engine.encoder.vectors = [_vector(1)]
    first, later = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert previous.speaker_id not in engine.diarizer._voices
    assert first.speaker_id is None
    assert first.boundary_reason == "identity_unavailable"
    assert first.track_id == previous.track_id
    assert first.track_reason == "boundary_continuity"
    assert later.track_id != first.track_id


def test_unmatched_long_turn_waits_for_independent_confirmation(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0), _vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    assert first.speaker_id is None and first.track_id is None
    assert first.reason == "pending_voice"
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert second.speaker_id and second.track_id
    assert len(engine.diarizer._voices) == 1
    (revision,) = engine.diarizer.revisions
    assert revision.window_index == 0
    assert (revision.start_sample, revision.end_sample) == (
        first.start_sample,
        first.end_sample,
    )
    assert (revision.speaker_id, revision.track_id) == (
        second.speaker_id,
        second.track_id,
    )
    assert len(engine.encoder.lengths) == 2
    assert engine.calls == 2
    assert not engine.diarizer._pending


def test_recent_short_turn_is_resolved_by_later_independent_long_turn(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 1.8, 1))
    engine.encoder.vectors = [_vector(0)] * 2
    short = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    assert short.speaker_id is None
    engine.scores = _scores((1, 4, 2))
    known = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert known.speaker_id is not None
    assert {item.window_index for item in engine.diarizer.revisions} == {0}
    assert all(
        item.speaker_id == known.speaker_id for item in engine.diarizer.revisions
    )
    assert len(engine.encoder.lengths) == 2  # No retrospective CAM or ASR.


def test_pending_two_people_keep_distinction_when_channels_permute(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1), (5, 9, 2))
    engine.encoder.vectors = [_vector(0), _vector(1), _vector(1), _vector(0)]
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert all(span.speaker_id is None for span in first)
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert second[0].speaker_id and second[1].speaker_id
    assert second[0].speaker_id != second[1].speaker_id
    changes = {item.start_sample: item for item in engine.diarizer.revisions}
    assert changes[first[0].start_sample].speaker_id == second[1].speaker_id
    assert changes[first[1].start_sample].speaker_id == second[0].speaker_id


def test_model_missed_change_on_same_channel_cannot_confirm_one_person(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1), (6, 9, 1))
    engine.encoder.vectors = [_vector(0), _vector(1)] * 2
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert all(span.speaker_id is None for span in first)
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert second[0].speaker_id != second[1].speaker_id
    assert all(span.speaker_id is not None for span in second)
    assert len(engine.diarizer._voices) == 2


def test_real_new_person_is_confirmed_without_fixed_person_count(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1))
    ids = []
    for index in range(3):
        engine.encoder.vectors = [_vector(index)] * 2
        assert (
            engine.diarizer.segment(_pcm(), stream_id="a", window_index=index * 2)[
                0
            ].speaker_id
            is None
        )
        ids.append(
            engine.diarizer.segment(_pcm(), stream_id="a", window_index=index * 2 + 1)[
                0
            ].speaker_id
        )
    assert None not in ids and len(set(ids)) == 3


def test_confirmed_moderate_long_references_help_without_chained_drift(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 4, 1))

    def angled(x: float) -> np.ndarray:
        return _vector(0) * x + _vector(1) * np.sqrt(1 - x * x)

    vectors = [
        _vector(0),
        _vector(0),
        angled(0.66),
        angled(0.67),
        angled(0.4),
        angled(-0.2),
    ]
    engine.encoder.vectors = vectors.copy()
    spans = [
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=index)[0]
        for index in range(6)
    ]
    identity = spans[1].speaker_id
    assert identity is not None
    assert all(span.speaker_id == identity for span in spans[2:5])
    assert spans[5].speaker_id is None
    voice = engine.diarizer._voices[identity]
    assert len(voice.members) == 3  # Confirmation plus the .66/.67 references.
    assert not any(np.array_equal(item, vectors[4]) for item in voice.members)
    assert np.dot(vectors[5], vectors[4]) > 0.8
    assert engine.diarizer._scores_for(vectors[5])[0][0] < 0.65


def test_pending_window_gap_and_expiration_do_not_connect_missing_audio(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 3
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    after_gap = engine.diarizer.segment(_pcm(), stream_id="a", window_index=2)[0]
    assert after_gap.speaker_id is None
    assert not engine.diarizer.revisions
    for index in range(3, 10):
        engine.diarizer.observe_silence(stream_id="a", window_index=index)
    assert not engine.diarizer._pending
    assert (
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=10)[0].speaker_id
        is None
    )


def test_complete_silence_preserves_person_but_not_pcm_continuity(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 3
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    known = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    engine.diarizer.observe_silence(stream_id="a", window_index=2)
    assert not engine.diarizer._boundary._pcm
    again = engine.diarizer.segment(_pcm(), stream_id="a", window_index=3)[0]
    assert (again.speaker_id, again.track_id) == (known.speaker_id, known.track_id)
    assert engine.calls == 3


def test_retrospective_revision_transaction_and_repeat_are_atomic(
    raw_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 3
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    before_pending = engine.diarizer._pending
    before_recent = engine.diarizer._recent.copy()
    original = engine.diarizer._reconcile_recent

    def fail(index: int) -> Any:
        original(index)
        raise RuntimeError("after retrospective changes")

    monkeypatch.setattr(engine.diarizer, "_reconcile_recent", fail)
    with pytest.raises(RuntimeError, match="retrospective"):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert not engine.diarizer._voices and not engine.diarizer.revisions
    assert engine.diarizer._pending is before_pending
    assert engine.diarizer._recent == before_recent
    monkeypatch.setattr(engine.diarizer, "_reconcile_recent", original)
    result = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    revisions = engine.diarizer.revisions
    assert revisions
    assert engine.diarizer.segment(_pcm(), stream_id="a", window_index=1) is result
    assert engine.diarizer.revisions is revisions


def test_stream_reset_clears_pending_revisions_and_voice_aliases(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _uncertain_scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 3
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert engine.diarizer.revisions
    changed = engine.diarizer.segment(_pcm(), stream_id="b", window_index=0)[0]
    assert changed.speaker_id is None and not engine.diarizer.revisions
    assert not engine.diarizer.track_aliases and not engine.diarizer._voice_aliases


def test_overlap_is_never_relabelled_by_retrospective_confirmation(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((0, 1, 4), (2, 5, 1))
    engine.encoder.vectors = [_vector(0)] * 2
    old = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    engine.scores = _scores((2, 5, 1))
    engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)
    assert all(
        change.start_sample != old[0].start_sample
        for change in engine.diarizer.revisions
    )
    assert engine.diarizer._recent[0].spans[0].overlap
    assert engine.diarizer._recent[0].spans[0].speaker_id is None


def _install_merge_candidates(diarizer: Any) -> tuple[str, str]:
    left = _vector(0)
    right = _vector(0) * 0.8 + _vector(1) * 0.6
    middle = _vector(0) * 0.9 + _vector(1) * np.sqrt(1 - 0.9**2)
    diarizer._voices = {
        "voice-01": speakers._Voice(left, left, 3, (left.copy(), middle.copy())),
        "voice-02": speakers._Voice(right, right, 3, (right.copy(), middle.copy())),
    }
    left_track = diarizer._tracks.assign(
        {0: left}, {0: 32000}, {0: 0}, {0: "voice-01"}, window_index=0
    )[0].track_id
    right_track = diarizer._tracks.assign(
        {0: right}, {0: 32000}, {0: 0}, {0: "voice-02"}, window_index=1
    )[0].track_id
    assert left_track is not None and right_track is not None
    return left_track, right_track


def test_mutually_supported_duplicate_voices_merge_with_revision_and_alias(
    raw_engine: Any,
) -> None:
    diarizer = raw_engine.diarizer
    left_track, right_track = _install_merge_candidates(diarizer)
    diarizer._recent = {
        0: speakers._RecentWindow(
            (speakers.SpeakerSpan(16000, 48000, "voice-02", track_id=right_track),),
            {},
            {0: 0},
        ),
        1: speakers._RecentWindow(
            (speakers.SpeakerSpan(32000, 64000, "voice-01", track_id=left_track),),
            {},
            {0: 0},
        ),
    }
    diarizer._reconcile_recent(1)
    assert list(diarizer._voices) == ["voice-01"]
    assert diarizer.track_aliases == ((right_track, left_track),)
    (revision,) = diarizer.revisions
    assert revision.window_index == 0 and revision.speaker_id == "voice-01"
    assert revision.track_id == left_track and revision.reason == "retrospective_merge"


def test_simultaneously_distinct_local_people_never_merge_despite_similar_voices(
    raw_engine: Any,
) -> None:
    diarizer = raw_engine.diarizer
    left_track, right_track = _install_merge_candidates(diarizer)
    diarizer._recent = {
        0: speakers._RecentWindow(
            (
                speakers.SpeakerSpan(0, 48000, "voice-01", track_id=left_track),
                speakers.SpeakerSpan(64000, 96000, "voice-02", track_id=right_track),
            ),
            {},
            {0: 0, 1: 1},
        ),
    }
    diarizer._reconcile_recent(0)
    assert len(diarizer._voices) == 2 and not diarizer.track_aliases
    # Clearing the short transcript cache must not erase the acoustic conflict.
    diarizer._recent.clear()
    diarizer._merge_voices()
    assert len(diarizer._voices) == 2


def test_reference_merges_cannot_walk_along_a_similarity_chain(raw_engine: Any) -> None:
    diarizer = raw_engine.diarizer

    def angle(degrees: float) -> np.ndarray:
        radians = np.deg2rad(degrees)
        return _vector(0) * np.cos(radians) + _vector(1) * np.sin(radians)

    left, bridge, right = angle(0), angle(35), angle(70)
    diarizer._voices = {
        "voice-01": speakers._Voice(left, left, 3, (bridge, angle(36)), (bridge,)),
        "voice-03": speakers._Voice(right, right, 3, (bridge.copy(), angle(36))),
    }
    assert (
        speakers.reference_score(left, right, diarizer._voices["voice-03"].members)
        > 0.8
    )
    assert (
        speakers.reference_score(right, left, diarizer._voices["voice-01"].members)
        > 0.8
    )
    diarizer._merge_voices()
    assert (
        len(diarizer._voices) == 2
    )  # Original endpoints conflict, despite the bridge.


def test_retrospective_matching_preserves_old_local_cannot_link(
    raw_engine: Any,
) -> None:
    diarizer = raw_engine.diarizer
    left_track, _ = _install_merge_candidates(diarizer)
    del diarizer._voices["voice-02"]
    unknown = speakers.SpeakerSpan(64000, 96000, None, clean_seconds=2)
    diarizer._recent = {
        0: speakers._RecentWindow(
            (
                speakers.SpeakerSpan(0, 48000, "voice-01", track_id=left_track),
                unknown,
            ),
            {1: _vector(0)},
            {0: 0, 1: 1},
        ),
    }
    result = diarizer._reconcile_recent(0)
    assert result[1].speaker_id is None and result[1].track_id is None


def test_recent_cache_and_references_remain_bounded(raw_engine: Any) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)] * 20
    for index in range(20):
        engine.diarizer.segment(_pcm(), stream_id="a", window_index=index)
    assert len(engine.diarizer._recent) == 6
    assert len(engine.diarizer._voices) == 1
    assert (
        len(next(iter(engine.diarizer._voices.values())).members)
        == speakers._MAX_REFERENCES
    )
    assert not engine.diarizer._pending


def test_reliable_distinct_long_voice_does_not_wait_for_another_window(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 4, 1))
    engine.encoder.vectors = [_vector(0)]
    span = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    assert span.speaker_id is not None and span.track_id is not None
    assert span.reason == "quality_anchor"
    assert not engine.diarizer._pending
    assert len(engine.encoder.lengths) == 1


def test_independently_matching_short_turns_can_confirm_without_long_speech(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 1.8, 1))
    engine.encoder.vectors = [_vector(0)] * 2
    first = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)[0]
    assert first.speaker_id is None and first.track_id is None
    second = engine.diarizer.segment(_pcm(), stream_id="a", window_index=1)[0]
    assert second.speaker_id is not None and second.track_id is not None
    assert second.reason == "confirmed_voice"
    (revision,) = engine.diarizer.revisions
    assert (revision.speaker_id, revision.track_id) == (
        second.speaker_id,
        second.track_id,
    )
    assert len(engine.encoder.lengths) == 2


def test_distinct_local_short_people_do_not_corroborate_each_other(
    raw_engine: Any,
) -> None:
    engine = raw_engine
    engine.scores = _scores((1, 1.8, 1), (3, 3.8, 2))
    engine.encoder.vectors = [_vector(0)] * 2
    spans = engine.diarizer.segment(_pcm(), stream_id="a", window_index=0)
    assert all(span.speaker_id is None and span.track_id is None for span in spans)
    assert not engine.diarizer._voices
