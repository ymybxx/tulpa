"""The owner adds sentences TA just missed; the runner learns them in place."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fish_memory.companion import owner_voice
from fish_memory.companion.runner import CompanionRunner


class Gate:
    def __init__(self, seconds):
        self.seconds = seconds
        self.verifier = None

    def unrecognized_seconds(self):
        return self.seconds

    def take_unrecognized(self):
        speech, self.seconds = b"\1\0" * int(self.seconds * 16_000), 0
        return speech, [speech]

    def replace_verifier(self, verifier):
        self.verifier = verifier


def runner(gate, published):
    value = CompanionRunner.__new__(CompanionRunner)
    value.settings = SimpleNamespace(
        owner_voice_dir="/owner",
        companion_audio_backend="livekit",
        owner_voice_threshold=0.5,
    )
    value._owner_voice_gate = gate
    value._owner_voice_verifier = None
    value._unrecognized_published = None
    value._unrecognized_expiry = None
    value._pet_events = SimpleNamespace(owner_voice_unrecognized=published.append)
    return value


@pytest.mark.asyncio
async def test_confirmed_missed_speech_extends_the_voiceprint_in_place(monkeypatch):
    added = []

    class Service:
        def __init__(self, directory):
            assert directory == Path("/owner")

        def add_speech(self, speech, *, audio_backend, heard):
            assert heard == [speech]
            added.append(len(speech))
            return {"enrollmentCount": 2}

        def load_verifier(self, *, audio_backend, threshold):
            return "extended"

    monkeypatch.setattr(owner_voice, "OwnerVoiceService", Service)
    published = []
    gate = Gate(3.5)
    value = runner(gate, published)
    await value._add_unrecognized_owner_voice()
    assert added == [3.5 * 32_000]
    assert gate.verifier == "extended" and value._owner_voice_verifier == "extended"
    assert published[-1] == {"seconds": 0.0, "enrollmentCount": 2}


@pytest.mark.asyncio
async def test_too_little_missed_speech_is_kept_and_explained(monkeypatch):
    monkeypatch.setattr(
        owner_voice, "OwnerVoiceService", lambda _: pytest.fail("nothing to add yet")
    )
    published = []
    gate = Gate(2.0)
    value = runner(gate, published)
    await value._add_unrecognized_owner_voice()
    value._unrecognized_expiry.cancel()
    assert gate.seconds == 2.0
    assert published[-1]["seconds"] == 2.0 and "不够" in published[-1]["error"]


@pytest.mark.asyncio
async def test_the_offer_goes_away_when_kept_sentences_are_forgotten(monkeypatch):
    import asyncio

    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _: real_sleep(0))
    published = []
    gate = Gate(3.5)
    value = runner(gate, published)
    value._publish_unrecognized_owner_voice(3.5)
    gate.seconds = 0  # forgotten after a few minutes of silence
    await asyncio.wait_for(value._unrecognized_expiry, timeout=1)
    assert published == [{"seconds": 3.5}, {"seconds": 0.0}]
