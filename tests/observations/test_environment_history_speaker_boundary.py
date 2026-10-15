from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fish_memory.observations.environment_history_speaker_boundary import (
    SpeakerBoundaryTracker,
)


@dataclass(frozen=True)
class _Turn:
    start: int
    end: int
    local_id: int | None
    overlap: bool = False


def _turn(start: float, end: float, local: int | None, overlap: bool = False) -> _Turn:
    return _Turn(int(start * 16_000), int(end * 16_000), local, overlap)


def _pcm(seconds: float = 10, value: int = 2048) -> bytes:
    return np.full(int(seconds * 16_000), value, dtype="<i2").tobytes()


def _remember(
    tracker: SpeakerBoundaryTracker,
    *,
    turns: tuple[_Turn, ...] = (_turn(8, 10, 0),),
    reliable: tuple[_Turn, ...] | None = None,
    identities: dict[int, str] | None = None,
    window_index: int = 3,
    stream_id: str = "movie",
    pcm: bytes | None = None,
) -> None:
    tracker.remember(
        _pcm() if pcm is None else pcm,
        stream_id=stream_id,
        window_index=window_index,
        turns=turns,
        reliable=turns if reliable is None else reliable,
        identities={0: "voice-a"} if identities is None else identities,
    )


class _Inference:
    def __init__(self, *turns: _Turn, error: bool = False) -> None:
        self.turns = turns
        self.error = error
        self.inputs: list[np.ndarray] = []

    def __call__(self, samples: np.ndarray) -> tuple[_Turn, ...]:
        self.inputs.append(samples.copy())
        if self.error:
            raise RuntimeError("temporary segmentation failure")
        return self.turns


def _resolve(
    tracker: SpeakerBoundaryTracker,
    infer: _Inference,
    *,
    turns: tuple[_Turn, ...] = (_turn(0, 0.7, 2),),
    reliable: tuple[_Turn, ...] | None = None,
    window_index: int = 4,
    stream_id: str = "movie",
    pcm: bytes | None = None,
):
    return tracker.resolve(
        _pcm(value=4096) if pcm is None else pcm,
        stream_id=stream_id,
        window_index=window_index,
        turns=turns,
        reliable=turns if reliable is None else reliable,
        infer=infer,
    )


def test_shared_time_regions_map_permuted_channels_to_previous_identity() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)  # Previous model call uses local channel zero.
    infer = _Inference(_turn(0, 2.7, 1))  # Joined model uses channel one.
    result = _resolve(tracker, infer)  # Current model uses channel two.
    assert result.proposals == {0: "voice-a"}
    assert result.reason == "continuous"
    assert len(infer.inputs) == 1
    audio = infer.inputs[0]
    assert audio.shape == (160_000,)
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio[:32_000], 2048 / 32768)
    np.testing.assert_array_equal(audio[32_000:64_000], 4096 / 32768)
    assert not audio[64_000:].any()
    # The helper produces only a new-window identity proposal, never old turns
    # or PCM to add to the ASR request, public transcript, or saved observation.
    assert list(result.proposals) == [0]


@pytest.mark.parametrize(
    "bridge",
    [
        (_turn(0, 2, 1), _turn(2, 2.7, 2)),  # A real change exactly at the seam.
        (_turn(0, 1.7, 1), _turn(2.1, 2.7, 1)),  # Same channel, long silence.
        (_turn(0, 1.9, 1), _turn(1.9, 2.1, None), _turn(2.1, 2.7, 1)),
        (_turn(0, 1.9, 1), _turn(1.9, 2.1, None, True), _turn(2.1, 2.7, 1)),
        (_turn(0, 2.2, 1), _turn(2.2, 2.7, 2)),  # Current source missed change.
    ],
)
def test_change_silence_uncertainty_or_overlap_at_join_never_inherits(
    bridge: tuple[_Turn, ...],
) -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    result = _resolve(tracker, _Inference(*bridge))
    assert result.proposals == {}
    assert result.reason == "boundary_unconfirmed"


def test_separate_unknown_previous_turn_cannot_borrow_earlier_known_identity() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker, turns=(_turn(8, 9.3, 0), _turn(9.3, 10, 1)))
    infer = _Inference(_turn(0, 2.7, 0))
    assert _resolve(tracker, infer).reason == "no_boundary_candidate"
    assert not infer.inputs


@pytest.mark.parametrize(
    ("previous", "identities"),
    [
        ((_turn(8, 10, None),), {}),
        ((_turn(8, 10, 0),), {}),
        ((_turn(8, 10, 0, True),), {0: "voice-a"}),
        ((_turn(8, 9.7, 0),), {0: "voice-a"}),
    ],
)
def test_unknown_overlapping_or_nonadjacent_previous_speech_skips_inference(
    previous: tuple[_Turn, ...], identities: dict[int, str]
) -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker, turns=previous, identities=identities)
    infer = _Inference(_turn(0, 2.7, 1))
    assert _resolve(tracker, infer).proposals == {}
    assert not infer.inputs


@pytest.mark.parametrize(
    "current",
    [
        (),
        (_turn(0, 0.7, None),),
        (_turn(0, 0.7, 2, True),),
        (_turn(0.3, 0.8, 2),),
        (_turn(0, 0.12, 2),),
    ],
)
def test_missing_uncertain_overlapping_or_short_current_head_skips_inference(
    current: tuple[_Turn, ...],
) -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    infer = _Inference(_turn(0, 2.7, 1))
    assert _resolve(tracker, infer, turns=current).proposals == {}
    assert not infer.inputs


def test_reference_confidence_dip_at_seam_cannot_propagate_identity() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker, reliable=(_turn(8, 9.7, 0), _turn(9.7, 10, None)))
    infer = _Inference(_turn(0, 2.7, 1))
    assert _resolve(tracker, infer).reason == "insufficient_evidence"
    assert not infer.inputs
    _remember(tracker)
    assert (
        _resolve(
            tracker, infer, reliable=(_turn(0, 0.3, None), _turn(0.3, 0.7, 2))
        ).reason
        == "insufficient_evidence"
    )
    assert not infer.inputs


def test_contradictory_known_previous_identity_rejects_bridge() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(
        tracker,
        turns=(_turn(8, 9.2, 0), _turn(9.2, 10, 1)),
        identities={0: "voice-a", 1: "voice-b"},
    )
    # The joined segmentation failed to keep the two previously verified people
    # apart. This is conflicting evidence, not permission to relabel the head.
    result = _resolve(tracker, _Inference(_turn(0, 2.7, 0)))
    assert result.reason == "boundary_unconfirmed"
    assert not result.proposals


def test_only_first_island_can_inherit_even_if_later_channel_repeats() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    result = _resolve(
        tracker,
        _Inference(_turn(0, 2.7, 1)),
        turns=(_turn(0, 0.7, 2), _turn(4, 8, 2)),
    )
    assert result.proposals == {0: "voice-a"}


def test_inference_failure_keeps_previous_cache_and_returns_no_proposal() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    infer = _Inference(error=True)
    assert _resolve(tracker, infer).reason == "inference_failed"
    assert len(infer.inputs) == 1
    assert _resolve(tracker, _Inference(_turn(0, 2.7, 1))).proposals == {0: "voice-a"}


@pytest.mark.parametrize(
    ("stream_id", "index"), [("movie", 3), ("movie", 2), ("movie", 5), ("other", 4)]
)
def test_gap_reorder_or_stream_change_cannot_consume_stale_tail(
    stream_id: str, index: int
) -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    infer = _Inference(_turn(0, 2.7, 1))
    result = _resolve(tracker, infer, stream_id=stream_id, window_index=index)
    assert result.reason == "not_contiguous"
    assert not infer.inputs
    tracker.clear()
    assert not tracker._pcm
    assert _resolve(tracker, infer).reason == "no_previous"


def test_cache_is_bounded_and_successful_remember_replaces_prior_tail() -> None:
    tracker = SpeakerBoundaryTracker()
    for index in range(30):
        _remember(tracker, window_index=index, pcm=_pcm(value=index))
        assert len(tracker._pcm) == 64_000
        assert len(tracker._turns) == 1
    infer = _Inference(_turn(0, 2.7, 1))
    assert _resolve(tracker, infer, window_index=30).proposals == {0: "voice-a"}
    np.testing.assert_array_equal(infer.inputs[0][:32_000], 29 / 32768)


def test_partial_previous_window_is_not_stitched_over_missing_audio() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    _remember(tracker, pcm=_pcm(9.5), window_index=4)
    infer = _Inference(_turn(0, 2.7, 1))
    assert not tracker._pcm
    assert _resolve(tracker, infer, window_index=5).reason == "no_previous"
    assert not infer.inputs


def test_partial_current_head_ignores_model_predictions_on_padding() -> None:
    tracker = SpeakerBoundaryTracker()
    _remember(tracker)
    infer = _Inference(_turn(0, 10, 1))
    result = _resolve(tracker, infer, pcm=_pcm(0.3), turns=(_turn(0, 0.3, 2),))
    assert result.proposals == {0: "voice-a"}
    assert not infer.inputs[0][36_800:].any()
    short = _resolve(tracker, infer, pcm=_pcm(0.1), turns=(_turn(0, 0.7, 2),))
    assert not short.proposals
    assert len(infer.inputs) == 1
