from uuid import UUID

from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.protocol import pack_pcm_frame

GENERATION = UUID("00000000-0000-0000-0000-000000000123")


def test_new_subscriber_recovers_latest_control_snapshot() -> None:
    hub = PetEventHub()
    hub.publisher_connected()
    hub.publish_control("pet.state", GENERATION, {"state": "thinking"})
    hub.publish_control(
        "pet.expression",
        GENERATION,
        {"emotion": "happy"},
    )

    snapshot = hub.snapshot()

    assert snapshot["generation"] == str(GENERATION)
    payload = snapshot["payload"]
    assert isinstance(payload, dict)
    assert payload["publisherConnected"] is True
    events = payload["events"]
    assert isinstance(events, list)
    assert [event["type"] for event in events] == [
        "pet.state",
        "pet.expression",
    ]


def test_slow_subscriber_drops_old_pcm_before_control() -> None:
    hub = PetEventHub(subscriber_queue_size=2)
    _subscriber_id, queue = hub.subscribe()
    first = pack_pcm_frame(GENERATION, 0, b"\0\0")
    second = pack_pcm_frame(GENERATION, 1, b"\0\0")
    hub.publish_pcm(first)
    hub.publish_pcm(second)

    hub.publish_control("pet.audio.stop", GENERATION, {})

    buffered = [queue.get_nowait(), queue.get_nowait()]
    assert [item.kind for item in buffered] == ["pcm", "control"]
    assert buffered[0].value == second
