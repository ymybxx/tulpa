from uuid import UUID

import pytest

from fish_memory.pet.protocol import (
    PET_PROTOCOL_VERSION,
    pack_pcm_frame,
    parse_published_control,
    unpack_pcm_frame,
)

GENERATION = UUID("00000000-0000-0000-0000-000000000123")


def test_pcm_frame_round_trip_keeps_generation_and_sample_position() -> None:
    frame = pack_pcm_frame(GENERATION, 640, b"\x01\x00\x02\x00")

    generation, position, pcm = unpack_pcm_frame(frame)

    assert frame[0] == PET_PROTOCOL_VERSION
    assert generation == GENERATION
    assert position == 640
    assert pcm == b"\x01\x00\x02\x00"


def test_control_protocol_rejects_unknown_event() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_published_control({"type": "pet.unknown", "payload": {}})


def test_control_protocol_accepts_nullable_generation() -> None:
    parsed = parse_published_control(
        {
            "type": "pet.state",
            "generation": None,
            "payload": {"state": "standby"},
        }
    )

    assert parsed.generation is None
    assert parsed.payload == {"state": "standby"}


def test_control_protocol_rejects_invalid_expression() -> None:
    with pytest.raises(ValueError, match="emotion"):
        parse_published_control(
            {
                "type": "pet.expression",
                "generation": str(GENERATION),
                "payload": {"emotion": "confused"},
            }
        )


def test_control_protocol_rejects_removed_gesture_field() -> None:
    with pytest.raises(ValueError, match="gesture"):
        parse_published_control(
            {
                "type": "pet.expression",
                "generation": str(GENERATION),
                "payload": {"emotion": "happy", "gesture": "nod"},
            }
        )


def test_control_protocol_accepts_hearing_text_and_rejects_extra_fields() -> None:
    parsed = parse_published_control(
        {"type": "pet.voice.hearing", "generation": None, "payload": {"text": "你好"}}
    )
    assert parsed.payload == {"text": "你好"}

    with pytest.raises(ValueError, match="hearing"):
        parse_published_control(
            {
                "type": "pet.voice.hearing",
                "generation": None,
                "payload": {"text": "你好", "fishId": "x"},
            }
        )
