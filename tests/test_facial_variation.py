import random

from fish_memory.facial_test.core import Channel, decode_jev, varied_level

MOUTH = Channel(
    description="嘴角",
    anchors=(-1.0, -0.4, 0.0, 0.4, 1.0),
    levels=("明显下垂", "微微下垂", "平常", "微微上扬", "明显上扬"),
    neutral=0.0,
)


def answer(score, probabilities):
    return {
        "type": "score",
        "score": score,
        "probabilities": {str(k): v for k, v in probabilities.items()},
    }


def test_a_confident_answer_is_used_as_scored():
    rng = random.Random(1)
    assert varied_level(answer(0.05, {0: 0.95, 1: 0.05}), MOUTH, rng) is None


def test_an_unsure_frown_frowns_more_or_less_but_never_smiles():
    rng = random.Random(2)
    unsure = answer(1.0, {0: 0.45, 1: 0.3, 2: 0.05, 3: 0.15, 4: 0.05})
    drawn = {varied_level(unsure, MOUTH, rng) for _ in range(1000)}
    assert drawn == {0, 1}


def test_an_unsure_usual_step_leans_only_the_way_the_model_leaned():
    rng = random.Random(3)
    unsure = answer(2.3, {1: 0.1, 2: 0.5, 3: 0.4})
    drawn = {varied_level(unsure, MOUTH, rng) for _ in range(1000)}
    assert drawn == {2, 3}


def test_without_rng_the_face_is_exactly_as_scored():
    raw = {"answers": {"mouth": answer(1.5, {0: 0.3, 1: 0.4, 2: 0.3})}}
    assert decode_jev(raw, {"mouth": MOUTH}) == {"mouth": -0.2}
    varied = decode_jev(raw, {"mouth": MOUTH}, random.Random(4))
    assert varied["mouth"] in {-1.0, -0.4, 0.0}
