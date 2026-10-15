import asyncio
from array import array
from types import SimpleNamespace

import pytest

import fish_memory.companion.audio as audio_module
from fish_memory.companion.audio import (
    PcmPlayback,
    PlayedOffsetTracker,
    SoundDeviceFullDuplex,
    SoundDevicePcmInput,
)
from fish_memory.companion.playback import PlaybackArbiter


def test_display_offset_uses_the_end_of_the_current_audio_segment() -> None:
    tracker = PlayedOffsetTracker()
    tracker.add_segment(byte_length=10, text_length=4)
    tracker.add_segment(byte_length=6, text_length=2)

    assert tracker.display_offset == 0

    tracker.consume(5)
    assert tracker.display_offset == 4

    tracker.consume(5)
    assert tracker.display_offset == 4

    tracker.consume(3)
    assert tracker.display_offset == 6


def test_display_offset_tracks_absolute_text_ranges() -> None:
    tracker = PlayedOffsetTracker()
    tracker.add_segment(byte_length=10, start_offset=8, end_offset=12)

    tracker.consume(5)
    assert tracker.display_offset == 12


def test_display_offset_can_resume_from_an_absolute_initial_offset() -> None:
    tracker = PlayedOffsetTracker(initial_offset=8)

    assert tracker.display_offset == 8
    tracker.add_segment(byte_length=10, start_offset=8, end_offset=12)
    assert tracker.display_offset == 8
    tracker.consume(1)
    assert tracker.display_offset == 12

    tracker.consume(5)
    assert tracker.display_offset == 12


def test_a_long_merged_sentence_is_revealed_as_it_is_heard() -> None:
    tracker = PlayedOffsetTracker()
    # A short first sentence shows whole and teaches the rate: 10 bytes a char.
    tracker.add_segment(byte_length=170, start_offset=0, end_offset=17)
    tracker.mark(17)
    tracker.consume(1)
    assert tracker.display_offset == 17
    tracker.consume(169)
    # The provider merged the next 104 characters into one sentence; only part
    # of its audio has arrived, so its length is judged from that rate.
    tracker.add_segment(byte_length=100, start_offset=17, end_offset=121)
    tracker.consume(1)
    assert tracker.display_offset == 18
    tracker.consume(99)
    assert tracker.display_offset == 17 + 10
    tracker.add_segment(byte_length=940, start_offset=17, end_offset=121)
    tracker.mark(121)
    # All of its audio is in: the share heard is exact.
    tracker.consume(420)
    assert tracker.display_offset == 17 + 52
    tracker.consume(520)
    assert tracker.played_offset == tracker.display_offset == 121


def test_a_long_first_sentence_shows_whole_until_its_length_is_known() -> None:
    tracker = PlayedOffsetTracker()
    tracker.add_segment(byte_length=100, start_offset=0, end_offset=60)
    tracker.consume(1)
    assert tracker.display_offset == 60


def test_display_offset_never_exceeds_queued_text() -> None:
    tracker = PlayedOffsetTracker()
    tracker.add_segment(byte_length=4, text_length=3)

    tracker.consume(100)

    assert tracker.display_offset == 3


def test_audio_byte_progress_counts_rendered_effects_without_advancing_text():
    tracker = PlayedOffsetTracker(initial_offset=7)
    tracker.add_segment(byte_length=8, start_offset=7, end_offset=7)
    tracker.mark(7)
    tracker.consume(4)
    assert tracker.consumed_bytes == 4
    assert tracker.played_offset == tracker.display_offset == 7
    tracker.consume(20)
    assert tracker.consumed_bytes == 8
    assert tracker.played_offset == tracker.display_offset == 7
    tracker.consume(10)
    assert tracker.consumed_bytes == 8


@pytest.mark.asyncio
async def test_sounddevice_input_forwards_pcm_blocks_and_stops() -> None:
    callback_holder: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            callback = callback_holder["callback"]
            assert callable(callback)
            callback(b"\x01\x02", 2, None)

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    stop_event = asyncio.Event()
    blocks: list[bytes] = []

    async def on_pcm(pcm: bytes) -> None:
        blocks.append(pcm)
        stop_event.set()

    audio = SoundDevicePcmInput(stream_factory=FakeStream)
    await audio.run(on_pcm, stop_event)

    assert blocks == [b"\x01\x02"]


@pytest.mark.asyncio
async def test_pcm_playback_reset_clears_pending_audio_and_played_offset() -> None:
    playback = PcmPlayback()
    await playback.append(b"\x00\x00" * 4, start_offset=0, end_offset=4)
    playback._callback(bytearray(4), 2)
    assert playback.display_offset == 4
    assert playback.played_audio_bytes == 4

    playback.reset()

    assert playback.display_offset == 0
    assert playback.played_audio_bytes == 0


@pytest.mark.asyncio
async def test_pcm_playback_preserves_owner_start_and_render_callbacks() -> None:
    callback_holder: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    playback = PcmPlayback(sample_rate=16_000, stream_factory=FakeStream)
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("answer", None)
    started: list[object] = []
    rendered: list[bytes] = []
    playback.set_playback_started_callback(started.append)
    playback.set_rendered_audio_callback(rendered.append)
    await playback.start()
    await playback.append_owned(
        owner,
        b"r" * 320,
        start_offset=3,
        end_offset=8,
    )

    callback = callback_holder["callback"]
    assert callable(callback)
    output = bytearray(320)
    callback(output, 160, None, None)
    await asyncio.sleep(0)

    assert output == b"r" * 320
    assert playback.display_offset == 8
    assert started == [owner]
    assert rendered == [b"r" * 320]
    await asyncio.wait_for(playback.wait_until_owner_drained(owner), timeout=0.1)
    await playback.stop()


@pytest.mark.asyncio
async def test_pcm_playback_accepts_explicit_low_latency_stream_shape() -> None:
    options: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            options.update(kwargs)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    playback = PcmPlayback(
        sample_rate=16_000,
        block_size=160,
        latency="low",
        stream_factory=FakeStream,
    )

    await playback.start()

    assert options["samplerate"] == 16_000
    assert options["blocksize"] == 160
    assert options["latency"] == "low"
    assert options["channels"] == 1
    assert options["dtype"] == "int16"
    assert callable(options["callback"])
    await playback.stop()


@pytest.mark.asyncio
async def test_pcm_playback_can_restart_after_physical_stream_stop() -> None:
    streams: list[object] = []

    class FakeStream:
        def __init__(self, **_kwargs: object) -> None:
            self.started = 0
            self.stopped = 0
            self.closed = 0
            streams.append(self)

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stopped += 1

        def close(self) -> None:
            self.closed += 1

    playback = PcmPlayback(sample_rate=16_000, stream_factory=FakeStream)

    await playback.start()
    await playback.start()
    await playback.stop()
    await playback.start()
    await playback.stop()

    assert len(streams) == 2
    assert [stream.started for stream in streams] == [1, 1]
    assert [stream.stopped for stream in streams] == [1, 1]
    assert [stream.closed for stream in streams] == [1, 1]


@pytest.mark.asyncio
async def test_pcm_playback_cleans_up_failed_stream_before_retry() -> None:
    streams: list[object] = []

    class FakeStream:
        def __init__(self, **_kwargs: object) -> None:
            self.index = len(streams)
            self.stopped = 0
            self.closed = 0
            streams.append(self)

        def start(self) -> None:
            if self.index == 0:
                raise RuntimeError("output unavailable")

        def stop(self) -> None:
            self.stopped += 1

        def close(self) -> None:
            self.closed += 1

    playback = PcmPlayback(sample_rate=16_000, stream_factory=FakeStream)

    with pytest.raises(RuntimeError, match="output unavailable"):
        await playback.start()
    await playback.start()
    await playback.stop()

    assert len(streams) == 2
    assert [stream.stopped for stream in streams] == [1, 1]
    assert [stream.closed for stream in streams] == [1, 1]


@pytest.mark.asyncio
async def test_pcm_playback_reset_releases_owner_without_false_start() -> None:
    playback = PcmPlayback(sample_rate=16_000)
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("agent", None)
    await playback.append_owned(
        owner,
        b"r" * 320,
        start_offset=0,
        end_offset=4,
    )

    playback.reset(2)

    assert playback.display_offset == 2
    await asyncio.wait_for(playback.wait_until_owner_drained(owner), timeout=0.1)


@pytest.mark.asyncio
async def test_full_duplex_pairs_capture_with_the_actual_rendered_frame() -> None:
    callback_holder: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    stop_event = asyncio.Event()
    frames: list[tuple[bytes, bytes]] = []

    async def on_frame(capture_pcm: bytes, render_pcm: bytes) -> None:
        frames.append((capture_pcm, render_pcm))
        stop_event.set()

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    task = asyncio.create_task(audio.run(on_frame, stop_event))
    await asyncio.sleep(0)
    await audio.append(b"r" * 320, start_offset=0, end_offset=10)

    callback = callback_holder["callback"]
    assert callable(callback)
    outdata = bytearray(320)
    callback(b"c" * 320, outdata, 160, None, None)
    await asyncio.wait_for(task, timeout=1)

    assert frames == [(b"c" * 320, b"r" * 320)]
    assert audio.display_offset == 10


@pytest.mark.asyncio
async def test_full_duplex_raises_when_callback_stops_delivering_frames() -> None:
    stream_events: list[str] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            stream_events.append("start")

        def stop(self) -> None:
            stream_events.append("stop")

        def close(self) -> None:
            stream_events.append("close")

    audio = SoundDeviceFullDuplex(
        stream_factory=FakeStream,
        frame_watchdog_timeout_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="stopped delivering frames"):
        await asyncio.wait_for(
            audio.run(lambda *_: None, asyncio.Event()),
            timeout=1,
        )

    assert stream_events == ["start", "stop", "close"]
    assert audio._stream is None


@pytest.mark.asyncio
async def test_full_duplex_close_still_closes_when_stop_fails() -> None:
    stream_events: list[str] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            stream_events.append("start")

        def stop(self) -> None:
            stream_events.append("stop")
            raise RuntimeError("stop failed")

        def close(self) -> None:
            stream_events.append("close")

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    await audio.start()

    with pytest.raises(RuntimeError, match="stop failed"):
        await audio.close()

    assert stream_events == ["start", "stop", "close"]
    assert audio._stream is None


@pytest.mark.asyncio
async def test_full_duplex_close_reports_stop_and_close_failures() -> None:
    stream_events: list[str] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            stream_events.append("start")

        def stop(self) -> None:
            stream_events.append("stop")
            raise RuntimeError("stop failed")

        def close(self) -> None:
            stream_events.append("close")
            raise RuntimeError("close failed")

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    await audio.start()

    with pytest.raises(BaseExceptionGroup) as raised:
        await audio.close()

    assert [str(error) for error in raised.value.exceptions] == [
        "stop failed",
        "close failed",
    ]
    assert stream_events == ["start", "stop", "close"]
    assert audio._stream is None


@pytest.mark.asyncio
async def test_full_duplex_reports_actual_playback_once_per_reset() -> None:
    callback_holder: dict[str, object] = {}
    playback_events: list[str] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    stop_event = asyncio.Event()
    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    audio.set_playback_started_callback(
        lambda _owner: playback_events.append("started")
    )

    async def on_frame(*_: object) -> None:
        return None

    task = asyncio.create_task(audio.run(on_frame, stop_event))
    await asyncio.sleep(0)
    await audio.append(b"r" * 640, start_offset=0, end_offset=10)

    callback = callback_holder["callback"]
    assert callable(callback)
    callback(b"c" * 320, bytearray(320), 160, None, None)
    callback(b"c" * 320, bytearray(320), 160, None, None)
    await asyncio.sleep(0)

    assert playback_events == ["started"]

    audio.reset()
    await audio.append(b"r" * 320, start_offset=0, end_offset=5)
    callback(b"c" * 320, bytearray(320), 160, None, None)
    await asyncio.sleep(0)
    assert playback_events == ["started", "started"]

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_full_duplex_first_frame_callback_carries_playback_owner() -> None:
    callback_holder: dict[str, object] = {}
    playback_owners: list[object] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    audio.set_playback_started_callback(playback_owners.append)
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("agent", None)
    sink = arbiter.bind_pcm(owner, audio)
    stop_event = asyncio.Event()

    async def handle_frame(*_: object) -> None:
        return None

    task = asyncio.create_task(audio.run(handle_frame, stop_event))
    await asyncio.sleep(0)

    sink.reset(6)
    await sink.append(b"r" * 320, start_offset=6, end_offset=9)
    callback = callback_holder["callback"]
    assert callable(callback)
    callback(b"c" * 320, bytearray(320), 160, None, None)
    await asyncio.sleep(0)

    assert playback_owners == [owner]
    assert audio.display_offset == 9
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_first_frame_uses_owner_of_first_byte_when_frame_crosses_queues() -> None:
    audio = SoundDeviceFullDuplex()
    audio._loop = asyncio.get_running_loop()
    reported: list[object] = []
    audio.set_playback_started_callback(reported.append)
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("answer", None)

    await audio.append(b"a" * 160, start_offset=0, end_offset=1)
    await audio.append_owned(
        owner,
        b"b" * 160,
        start_offset=1,
        end_offset=2,
    )
    audio._callback(b"c" * 320, bytearray(320), 160, None, None)
    await asyncio.sleep(0)

    assert reported == [None]
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_owner_drain_does_not_wait_for_next_owner_pcm() -> None:
    audio = SoundDeviceFullDuplex()
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", None)
    first_sink = arbiter.bind_pcm(first, audio)
    await first_sink.append(b"a" * 320, start_offset=0, end_offset=1)

    first_drain = asyncio.create_task(first_sink.wait_until_drained())
    await asyncio.sleep(0)
    assert await arbiter.release(first) is True
    second = await arbiter.acquire("agent", None)
    second_sink = arbiter.bind_pcm(second, audio)
    await second_sink.append(b"b" * 320, start_offset=0, end_offset=1)

    first_output = bytearray(320)
    audio._callback(b"c" * 320, first_output, 160, None, None)
    await asyncio.wait_for(first_drain, timeout=0.1)

    assert first_output == b"a" * 320
    second_output = bytearray(320)
    audio._callback(b"c" * 320, second_output, 160, None, None)
    assert second_output == b"b" * 320
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_mirrors_only_pcm_written_to_the_output() -> None:
    callback_holder: dict[str, object] = {}
    mirrored: list[bytes] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            callback_holder["callback"] = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    audio.set_rendered_audio_callback(mirrored.append)
    stop_event = asyncio.Event()

    async def handle_frame(*_: object) -> None:
        return None

    task = asyncio.create_task(audio.run(handle_frame, stop_event))
    await asyncio.sleep(0)
    callback = callback_holder["callback"]
    assert callable(callback)

    callback(b"c" * 320, bytearray(320), 160, None, None)
    await audio.append(b"r" * 320, start_offset=0, end_offset=2)
    callback(b"c" * 320, bytearray(320), 160, None, None)
    await asyncio.sleep(0)

    assert mirrored == [b"r" * 320]
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_full_duplex_pause_mutes_render_without_consuming_queued_audio() -> None:
    audio = SoundDeviceFullDuplex()
    await audio.append(b"r" * 320, start_offset=0, end_offset=10)

    await audio.pause()
    muted = bytearray(320)
    audio._callback(b"c" * 320, muted, 160, None, None)

    assert muted == b"\0" * 320
    assert audio.display_offset == 0

    await audio.resume()
    rendered = bytearray(320)
    audio._callback(b"c" * 320, rendered, 160, None, None)

    assert rendered == b"r" * 320
    assert audio.display_offset == 10


@pytest.mark.asyncio
async def test_full_duplex_duck_reduces_volume_without_stopping_playback() -> None:
    audio = SoundDeviceFullDuplex(duck_gain=0.2)
    pcm = array("h", [10_000, -10_000] * 80).tobytes()
    await audio.append(pcm, start_offset=0, end_offset=10)

    await audio.duck()
    rendered = bytearray(320)
    audio._callback(b"c" * 320, rendered, 160, None, None)

    samples = array("h")
    samples.frombytes(rendered)
    assert samples[:2] == array("h", [2_000, -2_000])
    assert audio.display_offset == 10

    await audio.unduck()


@pytest.mark.asyncio
async def test_full_duplex_uses_10ms_mono_16khz_stream() -> None:
    captured: dict[str, object] = {}

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    stop_event = asyncio.Event()
    task = asyncio.create_task(audio.run(lambda *_: None, stop_event))
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert captured["samplerate"] == 16_000
    assert captured["channels"] == 1
    assert captured["blocksize"] == 160
    assert captured["dtype"] == "int16"
    assert captured["latency"] == "low"


@pytest.mark.asyncio
async def test_full_duplex_reports_total_input_and_output_stream_latency() -> None:
    class FakeStream:
        latency = (0.0125, 0.0345)

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)

    assert audio.stream_delay_ms is None
    await audio.start()

    assert audio.stream_delay_ms == 47
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_waits_for_dac_time_before_consuming_or_draining() -> None:
    audio = SoundDeviceFullDuplex()
    audio._loop = asyncio.get_running_loop()
    rendered: list[bytes] = []
    started: list[object] = []
    audio.set_rendered_audio_callback(rendered.append)
    audio.set_playback_started_callback(started.append)
    owner = await PlaybackArbiter().acquire("answer", None)
    await audio.append_owned(owner, b"r" * 320, start_offset=0, end_offset=4)
    await audio.mark(4)
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.0, outputBufferDacTime=1.04),
    )
    drain = asyncio.create_task(audio.wait_until_owner_drained(owner))
    await asyncio.sleep(0)
    assert audio.played_offset == 0 and audio.played_audio_bytes == 0
    assert not drain.done() and not rendered and not started
    # The entire frame ends at 1.05, not when it was submitted at 1.0.
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.06, outputBufferDacTime=1.10),
    )
    await asyncio.wait_for(drain, 0.1)
    assert audio.played_offset == 4 and audio.played_audio_bytes == 320
    assert rendered == [b"r" * 320] and started == [owner]
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_reset_discards_unconfirmed_device_audio() -> None:
    audio = SoundDeviceFullDuplex()
    await audio.append(b"r" * 320, start_offset=0, end_offset=4)
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.0, outputBufferDacTime=1.04),
    )
    audio.reset(10)
    await audio.append(b"s" * 320, start_offset=10, end_offset=14)
    await audio.mark(14)
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.06, outputBufferDacTime=1.10),
    )
    assert audio.played_offset == 10 and audio.played_audio_bytes == 0
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.12, outputBufferDacTime=1.16),
    )
    assert audio.played_offset == 14 and audio.played_audio_bytes == 320
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_discontinuity_never_confirms_pending_playback() -> None:
    audio = SoundDeviceFullDuplex()
    await audio.append(b"r" * 320, start_offset=0, end_offset=4)
    await audio.mark(4)
    audio._callback(
        bytes(320),
        bytearray(320),
        160,
        SimpleNamespace(currentTime=1.0, outputBufferDacTime=1.04),
    )
    out = bytearray(b"x" * 320)
    audio._callback(
        bytes(320),
        out,
        160,
        SimpleNamespace(currentTime=1.2, outputBufferDacTime=1.24),
        SimpleNamespace(output_underflow=True),
    )
    assert audio.played_offset == 0 and out == bytes(320)
    with pytest.raises(RuntimeError, match="discontinuity"):
        await audio.run(lambda *_: None, asyncio.Event(), start_if_needed=False)


@pytest.mark.asyncio
async def test_full_duplex_closes_failed_start_and_rejects_old_callbacks() -> None:
    streams: list[object] = []

    class FakeStream:
        def __init__(self, **options):
            self.callback = options["callback"]
            self.closed = False
            streams.append(self)

        def start(self):
            if len(streams) == 1:
                raise RuntimeError("device unavailable")

        def stop(self):
            pass

        def close(self):
            self.closed = True

    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    with pytest.raises(RuntimeError, match="device unavailable"):
        await audio.start()
    assert streams[0].closed and audio._stream is None
    await audio.start()
    await audio.append(b"r" * 320, start_offset=0, end_offset=4)
    output = bytearray(b"x" * 320)
    streams[0].callback(bytes(320), output, 160, None, None)
    assert output == bytes(320) and audio.played_audio_bytes == 0
    streams[1].callback(bytes(320), output, 160, None, None)
    assert output == b"r" * 320 and audio.played_audio_bytes == 320
    await audio.close()


def test_windows_audio_selects_wasapi_and_converts_16khz_mixer_format():
    sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": "MME"},
            {
                "name": "Windows WASAPI",
                "default_input_device": 3,
                "default_output_device": 4,
            },
        ],
        query_devices=lambda device, kind: {"hostapi": 1 if device in (3, 4) else 0},
        WasapiSettings=lambda **kwargs: kwargs,
    )
    assert audio_module._windows_stream_options(sd, None, None) == {
        "device": (3, 4),
        "extra_settings": ({"auto_convert": True}, {"auto_convert": True}),
    }
    explicit = audio_module._windows_stream_options(sd, 1, 2)
    assert explicit["device"] == (1, 2)
    assert explicit["extra_settings"] == (None, None)


@pytest.fixture
def windows_audio_devices(monkeypatch):
    streams = []
    defaults = {"input": -1, "output": 4}

    class FakeStream:
        active = True
        latency = 0.01

        def __init__(self, kind, **options):
            self.kind = kind
            self.options = options
            self.events = []
            streams.append(self)

        def start(self):
            self.events.append("start")

        def stop(self):
            self.events.append("stop")

        def close(self):
            self.events.append("close")

    sd = SimpleNamespace(
        query_hostapis=lambda: [
            {
                "name": "Windows WASAPI",
                "default_input_device": defaults["input"],
                "default_output_device": defaults["output"],
            }
        ],
        query_devices=lambda *_: {"hostapi": 0},
        WasapiSettings=lambda **options: options,
        RawStream=lambda **options: FakeStream("duplex", **options),
        RawOutputStream=lambda **options: FakeStream("output", **options),
    )
    monkeypatch.setattr(audio_module.sys, "platform", "win32")
    monkeypatch.setattr(audio_module, "import_module", lambda _: sd)
    return streams, defaults


@pytest.mark.asyncio
async def test_windows_muted_microphone_can_play_without_input_device(
    windows_audio_devices,
):
    streams, _ = windows_audio_devices
    audio = SoundDeviceFullDuplex()
    audio.set_microphone_enabled(False)
    rendered = []
    audio.set_rendered_audio_callback(rendered.append)
    await audio.start()
    assert streams[0].kind == "output"
    assert streams[0].options["device"] == 4
    assert streams[0].options["extra_settings"] == {"auto_convert": True}
    stop = asyncio.Event()
    observed = []
    received = asyncio.Event()

    async def on_frame(capture, render):
        observed.append((capture, render))
        received.set()

    running = asyncio.create_task(audio.run(on_frame, stop, start_if_needed=False))
    await asyncio.sleep(0)
    try:
        await audio.append(b"r" * 320, start_offset=0, end_offset=1)
        output = bytearray(320)
        callback = streams[0].options["callback"]
        callback(
            output,
            160,
            SimpleNamespace(
                currentTime=1.0,
                outputBufferDacTime=1.1,
            ),
            None,
        )
        assert output == b"r" * 320
        assert audio.played_audio_bytes == 0  # Still in the DAC queue.
        callback(
            bytearray(320),
            160,
            SimpleNamespace(
                currentTime=1.12,
                outputBufferDacTime=1.2,
            ),
            None,
        )
        await asyncio.wait_for(received.wait(), timeout=1)
        assert audio.played_audio_bytes == 320
        assert rendered == [b"r" * 320]
        assert observed and all(capture == bytes(320) for capture, _ in observed)
    finally:
        stop.set()
        await asyncio.wait_for(running, timeout=1)
    assert streams[0].events == ["start", "stop", "close"]


@pytest.mark.asyncio
async def test_windows_playback_only_still_requires_a_speaker(windows_audio_devices):
    streams, defaults = windows_audio_devices
    defaults["output"] = -1
    audio = SoundDeviceFullDuplex()
    audio.set_microphone_enabled(False)
    with pytest.raises(RuntimeError, match="speaker is unavailable"):
        await audio.start()
    assert not streams


@pytest.mark.asyncio
async def test_windows_unmuting_playback_only_reopens_real_capture(
    windows_audio_devices,
):
    streams, defaults = windows_audio_devices
    audio = SoundDeviceFullDuplex()
    audio.set_microphone_enabled(False)
    await audio.start()
    audio.set_microphone_enabled(True)
    with pytest.raises(RuntimeError, match="microphone.*reopen"):
        await audio.run(lambda *_: None, asyncio.Event(), start_if_needed=False)
    assert streams[0].events == ["start", "stop", "close"]
    with pytest.raises(RuntimeError, match="microphone or speaker is unavailable"):
        await audio.start()
    defaults["input"] = 3
    await audio.start()
    assert streams[-1].kind == "duplex"
    assert streams[-1].options["device"] == (3, 4)
    await audio.close()


@pytest.mark.asyncio
async def test_full_duplex_discards_pending_capture_frames_when_closed() -> None:
    audio = SoundDeviceFullDuplex()
    audio._enqueue_frame(b"c" * 320, b"r" * 320)

    await audio.close()

    assert audio._frames.empty()


@pytest.mark.asyncio
async def test_full_duplex_exposes_dequeued_frame_age_and_queue_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((10.0, 10.042))
    monkeypatch.setattr(audio_module, "monotonic", lambda: next(clock))

    class FakeStream:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    stop = asyncio.Event()
    observed: list[tuple[float, int, int]] = []
    audio = SoundDeviceFullDuplex(stream_factory=FakeStream)
    audio._enqueue_frame(b"c" * 320, b"r" * 320)
    assert audio.frame_queue_depth == 1

    async def on_frame(_capture: bytes, _render: bytes) -> None:
        observed.append(
            (
                audio.last_dequeued_frame_age_ms,
                audio.last_dequeued_frame_queue_depth,
                audio.frame_queue_depth,
            )
        )
        stop.set()

    await audio.run(on_frame, stop)

    assert observed == [(42.0, 0, 0)]


def test_full_duplex_queue_telemetry_tracks_depth_and_drops() -> None:
    audio = SoundDeviceFullDuplex()
    for _ in range(64):
        audio._enqueue_frame(b"c" * 320, b"r" * 320)

    assert audio.frame_queue_depth == 64
    assert audio.dropped_frames == 0
    audio._enqueue_frame(b"x" * 320, b"y" * 320)
    assert audio.frame_queue_depth == 64
    assert audio.dropped_frames == 1
