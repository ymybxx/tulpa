import asyncio
from uuid import uuid4

import pytest

from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.protocol import pack_pcm_frame, parse_published_control


def payload(**changes):
    return dict(
        fishId=str(uuid4()),
        sessionId=str(uuid4()),
        messageId=str(uuid4()),
        playbackId=str(uuid4()),
        revision=1,
        status="playing",
        displayOffset=4,
        consumedOffset=0,
        visibleContent="第一句。",
        **changes,
    )


@pytest.mark.asyncio
async def test_control_only_subscriber_gets_progress_snapshot_without_pcm():
    hub = PetEventHub()
    _, queue = hub.subscribe(control_only=True)
    value = payload(messageSequence=10027)
    event = hub.publish_control("pet.playback.progress", uuid4(), value)
    assert (await queue.get()).kind == "control"
    hub.publish_pcm(pack_pcm_frame(uuid4(), 0, b"\0\0"))
    assert queue.empty()
    assert event in hub.snapshot()["payload"]["events"]
    assert event["sequence"] != 10027
    assert event["payload"]["messageSequence"] == 10027
    hub.publish_control("pet.playback.progress", uuid4(), value)
    assert queue.empty()
    for _ in range(40):
        hub.publish_control("pet.playback.progress", uuid4(), payload())
    assert len(hub.snapshot()["payload"]["events"]) == 32


@pytest.mark.asyncio
async def test_page_interrupt_waits_for_companion_persistence_ack():
    hub = PetEventHub()
    _, queue = hub.subscribe(control_only=True)
    fish_id, message_id = uuid4(), uuid4()
    requested = asyncio.create_task(hub.request_answer_interrupt(fish_id, message_id))
    event = await queue.get()
    assert "pet.answer.interrupt" in event.value
    assert not requested.done()
    hub.answer_interrupted(message_id)
    await requested


@pytest.mark.parametrize(
    "changes",
    [
        dict(displayOffset=5),
        dict(consumedOffset=5),
        dict(revision=True),
        dict(status="unknown"),
        dict(messageSequence=True),
        dict(messageSequence=-1),
        dict(messageSequence="10027"),
    ],
)
def test_invalid_playback_progress_is_rejected(changes):
    value = payload()
    value.update(changes)
    with pytest.raises(ValueError):
        parse_published_control(
            dict(type="pet.playback.progress", generation=None, payload=value)
        )


def test_answer_progress_does_not_replace_the_avatar_audio_generation():
    hub = PetEventHub()
    avatar_generation = uuid4()
    hub.publish_control("pet.audio.start", avatar_generation, {"sampleRate": 16000})
    hub.publish_control("pet.playback.progress", uuid4(), payload())
    assert hub.snapshot()["generation"] == str(avatar_generation)


def test_terminal_progress_is_scoped_and_replaced_by_new_playback():
    hub = PetEventHub()
    fish_id, message_id = uuid4(), uuid4()
    old = payload()
    old.update(
        fishId=str(fish_id),
        messageId=str(message_id),
        status="completed",
        consumedOffset=4,
    )
    hub.publish_control("pet.playback.progress", None, old)
    assert hub.terminal_answer_progress(fish_id, message_id)["consumedOffset"] == 4
    assert hub.terminal_answer_progress(uuid4(), message_id) is None
    current = {**old, "playbackId": str(uuid4()), "status": "buffering", "revision": 1}
    hub.publish_control("pet.playback.progress", None, current)
    assert hub.terminal_answer_progress(fish_id, message_id) is None
