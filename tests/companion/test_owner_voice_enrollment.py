from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fish_memory.companion.owner_voice import OwnerVoiceError
from fish_memory.companion.owner_voice_enrollment import capture_owner_voice


class FakeAudio:
    capture_is_preprocessed = True
    dropped_frames = 0
    stream_delay_ms = 73

    def __init__(self):
        self.closed = 0
        self.started = False

    async def run(self, callback, stop):
        self.started = True
        while not stop.is_set():
            await callback(b"\x01\x02" * 160, bytes(320))

    async def close(self):
        self.closed += 1


@pytest.fixture
def settings():
    return SimpleNamespace(companion_aec_delay_ms=50, companion_aec_delay_mode="auto")


def use_audio(monkeypatch, audio):
    monkeypatch.setattr(
        "fish_memory.companion.runner._default_audio_backend",
        lambda settings: (audio, audio),
    )


@pytest.mark.asyncio
async def test_capture_announces_first_frame_and_closes_without_asr(
    settings, monkeypatch
):
    audio = FakeAudio()
    use_audio(monkeypatch, audio)
    calls = []

    def ready():
        assert audio.started
        calls.append("recording")

    captured = await capture_owner_voice(
        settings, duration_seconds=6, on_recording=ready
    )
    assert calls == ["recording"]
    assert captured == b"\x01\x02" * (6 * 16_000)
    assert audio.closed == 1


@pytest.mark.asyncio
async def test_capture_uses_same_aec_and_device_delay_for_unprocessed_backend(
    settings, monkeypatch
):
    audio = FakeAudio()
    audio.capture_is_preprocessed = False
    use_audio(monkeypatch, audio)
    calls = []

    class Aec:
        def __init__(self, **kwargs):
            assert kwargs == {"stream_delay_ms": 50, "delay_mode": "auto"}

        def set_stream_delay(self, delay):
            calls.append(delay)

        def process_reverse_stream(self, rendered):
            assert rendered == bytes(320)

        def process_stream(self, captured):
            assert captured == b"\x01\x02" * 160
            return b"\x02\x03" * 160

    monkeypatch.setattr("fish_memory.companion.audio_processing.WebRtcAec", Aec)
    captured = await capture_owner_voice(
        settings, duration_seconds=6, on_recording=lambda: None
    )
    assert captured == b"\x02\x03" * (6 * 16_000)
    assert calls == [73]
    assert audio.closed == 1


@pytest.mark.asyncio
async def test_capture_cancellation_closes_helper(settings, monkeypatch):
    audio = FakeAudio()
    use_audio(monkeypatch, audio)
    started = asyncio.Event()

    async def run(callback, stop):
        started.set()
        await asyncio.Event().wait()

    audio.run = run
    task = asyncio.create_task(
        capture_owner_voice(settings, duration_seconds=6, on_recording=lambda: None)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert audio.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "short", "dropped"])
async def test_capture_failures_close_helper_and_reject_audio(
    settings, monkeypatch, failure
):
    audio = FakeAudio()
    use_audio(monkeypatch, audio)
    if failure == "dropped":
        audio.dropped_frames = 1
    else:

        async def run(callback, stop):
            if failure == "startup":
                raise RuntimeError("helper failed")
            await callback(bytes(320), bytes(320))

        audio.run = run
    with pytest.raises((OwnerVoiceError, RuntimeError)):
        await capture_owner_voice(
            settings, duration_seconds=6, on_recording=lambda: None
        )
    assert audio.closed == 1
