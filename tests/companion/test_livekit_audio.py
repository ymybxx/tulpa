import asyncio
import logging
import struct
from types import SimpleNamespace
from uuid import uuid4

import pytest

from fish_memory.companion.livekit_audio import (
    CONTROL_DUCK,
    CONTROL_ENSURE_PROCESSED,
    CONTROL_FLUSH,
    CONTROL_MAGIC,
    CONTROL_PAUSE,
    CONTROL_RESET,
    CONTROL_RESUME,
    CONTROL_STOP,
    CONTROL_UNDUCK,
    FRAME_MAGIC,
    FRAME_PAYLOAD_BYTES,
    PCM_FRAME_BYTES,
    PLAYBACK_MAGIC,
    RENDER_MAGIC,
    RENDER_PAYLOAD_BYTES,
    LiveKitProcessedPcmInput,
)
from fish_memory.companion.playback import PlaybackArbiter, PlaybackOwner


def packet(
    sequence: int,
    *,
    captured_at_ns: int = 1_000_000_000,
    pcm: bytes | None = None,
    magic: bytes = FRAME_MAGIC,
    payload_length: int = FRAME_PAYLOAD_BYTES,
) -> bytes:
    audio = pcm if pcm is not None else bytes([sequence % 251]) * PCM_FRAME_BYTES
    return (
        struct.pack("<4sI", magic, payload_length)
        + struct.pack("<QQ", sequence, captured_at_ns)
        + audio
    )


def render_packet(
    generation: int,
    cumulative_valid_bytes: int,
    *,
    valid_byte_count: int = PCM_FRAME_BYTES,
    rendered_at_ns: int = 1_000_000_000,
    pcm: bytes | None = None,
) -> bytes:
    audio = pcm if pcm is not None else b"r" * PCM_FRAME_BYTES
    return (
        struct.pack("<4sI", RENDER_MAGIC, RENDER_PAYLOAD_BYTES)
        + struct.pack(
            "<QQQI",
            generation,
            cumulative_valid_bytes,
            rendered_at_ns,
            valid_byte_count,
        )
        + audio
    )


def decoded_packets(data: list[bytes]) -> list[tuple[bytes, bytes]]:
    result: list[tuple[bytes, bytes]] = []
    for value in data:
        magic, payload_length = struct.unpack_from("<4sI", value)
        assert len(value) == 8 + payload_length
        result.append((magic, value[8:]))
    return result


class FakeStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False
        self.write_error: BaseException | None = None
        self.drain_gate = asyncio.Event()
        self.drain_gate.set()

    def write(self, data: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        if self.closed:
            raise BrokenPipeError("stdin is closed")
        self.written.append(bytes(data))

    async def drain(self) -> None:
        await self.drain_gate.wait()

    def close(self) -> None:
        self.closed = True
        self.drain_gate.set()


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = FakeStdin()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode: int | None = None
        self._waited: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.terminate_calls = 0
        self.kill_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.exit(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.exit(-9)

    async def wait(self) -> int:
        return await self._waited

    def exit(self, code: int) -> None:
        if self.returncode is not None:
            return
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._waited.set_result(code)


def make_input(
    process: FakeProcess,
    **kwargs: object,
) -> tuple[
    LiveKitProcessedPcmInput,
    list[tuple[tuple[object, ...], dict[str, object]]],
]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def spawn(*args: object, **options: object) -> FakeProcess:
        calls.append((args, options))
        return process

    value = LiveKitProcessedPcmInput(
        "/fake/fish-memory-livekit-audio-helper",
        process_factory=spawn,
        **kwargs,
    )
    return value, calls


async def start_with_first_frame(
    audio: LiveKitProcessedPcmInput,
    process: FakeProcess,
    *,
    captured_at_ns: int = 1_000_000_000,
) -> None:
    task = asyncio.create_task(audio.start())
    await asyncio.sleep(0)
    process.stdout.feed_data(packet(0, captured_at_ns=captured_at_ns))
    await task


@pytest.mark.asyncio
async def test_livekit_input_parses_fixed_frames_and_reports_age() -> None:
    process = FakeProcess()
    audio, calls = make_input(process, clock_ns=lambda: 1_003_000_000)
    await start_with_first_frame(audio, process)
    assert audio.frame_queue_depth == 1
    observed: list[tuple[bytes, bytes]] = []
    stop = asyncio.Event()

    async def consume(capture_pcm: bytes, render_pcm: bytes) -> None:
        observed.append((capture_pcm, render_pcm))
        stop.set()

    await audio.run(consume, stop)

    assert calls[0][0] == ("/fake/fish-memory-livekit-audio-helper",)
    assert calls[0][1]["stdin"] == asyncio.subprocess.PIPE
    assert observed == [(bytes(PCM_FRAME_BYTES), bytes(PCM_FRAME_BYTES))]
    assert audio.last_dequeued_frame_age_ms == 3.0
    assert audio.last_dequeued_frame_queue_depth == 0
    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_livekit_input_counts_sequence_gaps_and_bounded_queue_drops() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, queue_size=2)
    await start_with_first_frame(audio, process)

    process.stdout.feed_data(packet(3))
    process.stdout.feed_data(packet(4))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert audio.frame_queue_depth == 2
    # Missing helper sequences 1 and 2, plus the oldest local queued frame.
    assert audio.dropped_frames == 3
    first, _, _ = audio._frames.get_nowait()
    second, _, _ = audio._frames.get_nowait()
    assert first == bytes([3]) * PCM_FRAME_BYTES
    assert second == bytes([4]) * PCM_FRAME_BYTES
    await audio.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (packet(0, magic=b"BAD!"), "packet magic"),
        (
            struct.pack("<4sI", FRAME_MAGIC, FRAME_PAYLOAD_BYTES - 1),
            "payload length",
        ),
        (packet(1), "sequence must be 0"),
    ],
)
async def test_livekit_input_rejects_invalid_first_packet(
    data: bytes,
    message: str,
) -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    task = asyncio.create_task(audio.start())
    await asyncio.sleep(0)
    process.stdout.feed_data(data)

    with pytest.raises(RuntimeError, match=message):
        await task
    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_livekit_input_reports_exit_and_drains_bounded_stderr() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    task = asyncio.create_task(audio.start())
    await asyncio.sleep(0)
    process.stderr.feed_data(b"native VPIO failed\n")
    await asyncio.sleep(0)
    process.exit(7)

    with pytest.raises(RuntimeError, match="stdout closed unexpectedly") as raised:
        await task
    assert "native VPIO failed" in str(raised.value)


@pytest.mark.asyncio
async def test_livekit_input_first_frame_timeout_is_a_hard_failure() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, startup_timeout_seconds=0.01)

    with pytest.raises(RuntimeError, match="first frame"):
        await audio.start()

    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_livekit_input_runtime_exit_is_a_hard_failure() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    process.exit(3)
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="stdout closed unexpectedly"):
        await audio.run(lambda *_: None, asyncio.Event())


@pytest.mark.asyncio
async def test_livekit_input_runtime_watchdog_is_a_hard_failure() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, frame_watchdog_timeout_seconds=0.01)
    await start_with_first_frame(audio, process)

    with pytest.raises(RuntimeError, match="stopped delivering frames"):
        await audio.run(lambda *_: None, asyncio.Event())


@pytest.mark.asyncio
async def test_livekit_input_rejects_duplicate_sequence_after_start() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    process.stdout.feed_data(packet(0))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="increase strictly"):
        await audio.run(lambda *_: None, asyncio.Event())


@pytest.mark.asyncio
async def test_livekit_input_close_is_idempotent() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)

    await audio.close()
    await audio.close()

    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_livekit_input_can_restart_with_a_fresh_sequence() -> None:
    processes = [FakeProcess(), FakeProcess()]
    spawn_calls = 0

    async def spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        nonlocal spawn_calls
        process = processes[spawn_calls]
        spawn_calls += 1
        return process

    audio = LiveKitProcessedPcmInput(
        "/fake/fish-memory-livekit-audio-helper",
        process_factory=spawn,
    )

    for process in processes:
        task = asyncio.create_task(audio.start())
        await asyncio.sleep(0)
        process.stdout.feed_data(packet(0))
        await task
        await audio.close()

    assert spawn_calls == 2
    assert [process.terminate_calls for process in processes] == [1, 1]


@pytest.mark.asyncio
async def test_livekit_input_kills_helper_that_ignores_termination() -> None:
    class StubbornProcess(FakeProcess):
        def terminate(self) -> None:
            self.terminate_calls += 1

    process = StubbornProcess()
    audio, _ = make_input(process, shutdown_timeout_seconds=0.01)
    await start_with_first_frame(audio, process)

    await audio.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == -9


@pytest.mark.asyncio
async def test_livekit_input_rejects_missing_helper_before_spawn(
    tmp_path,
) -> None:
    missing = tmp_path / "missing-livekit-helper"
    audio = LiveKitProcessedPcmInput(str(missing))

    with pytest.raises(RuntimeError, match="missing or not executable"):
        await audio.start()


def test_livekit_input_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="path"):
        LiveKitProcessedPcmInput(" ")
    with pytest.raises(ValueError, match="queue size"):
        LiveKitProcessedPcmInput("/helper", queue_size=0)
    with pytest.raises(ValueError, match="timeouts"):
        LiveKitProcessedPcmInput("/helper", startup_timeout_seconds=0)
    with pytest.raises(ValueError, match="writer queue size"):
        LiveKitProcessedPcmInput("/helper", writer_queue_size=0)
    with pytest.raises(ValueError, match="timeouts"):
        LiveKitProcessedPcmInput("/helper", drain_watchdog_timeout_seconds=0)
    with pytest.raises(ValueError, match="at least one PCM16 frame"):
        LiveKitProcessedPcmInput("/helper", max_inflight_playback_bytes=318)


async def settle() -> None:
    for _ in range(4):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_livekit_playback_uses_native_rendered_ack_for_owner_and_offset() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    audio.reset(6)
    owner = PlaybackOwner(uuid4(), "answer", uuid4())
    started: list[PlaybackOwner | None] = []
    rendered: list[bytes] = []
    audio.set_playback_started_callback(started.append)
    audio.set_rendered_audio_callback(rendered.append)
    pcm = b"a" * 640

    await audio.append_owned(owner, pcm, start_offset=6, end_offset=9)
    await settle()

    generation = audio.playback_generation
    assert decoded_packets(process.stdin.written)[-1] == (
        PLAYBACK_MAGIC,
        struct.pack("<Q", generation) + pcm,
    )
    assert audio.display_offset == 6
    assert audio.queued_playback_bytes == 640
    assert audio.rendered_playback_bytes == 0
    assert started == []
    assert rendered == []

    process.stdout.feed_data(render_packet(generation, 320))
    await settle()

    assert audio.display_offset == 9
    assert audio.rendered_playback_bytes == 320
    assert started == [owner]
    assert rendered == [b"r" * 320]

    process.stdout.feed_data(render_packet(generation, 640, pcm=b"s" * 320))
    await settle()
    await asyncio.wait_for(audio.wait_until_owner_drained(owner), timeout=0.1)

    assert started == [owner]
    assert rendered == [b"r" * 320, b"s" * 320]
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_partial_flush_waits_for_valid_rendered_bytes() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 322, start_offset=0, end_offset=4)
    generation = audio.playback_generation
    process.stdout.feed_data(render_packet(generation, 320))
    await settle()
    drain = asyncio.create_task(audio.wait_until_drained())
    await settle()

    assert decoded_packets(process.stdin.written)[-1] == (
        CONTROL_MAGIC,
        struct.pack("<BQ", CONTROL_FLUSH, generation),
    )
    assert not drain.done()
    process.stdout.feed_data(
        render_packet(
            generation,
            322,
            valid_byte_count=2,
            pcm=b"aa" + bytes(318),
        )
    )
    await asyncio.wait_for(drain, timeout=0.1)

    assert audio.rendered_playback_bytes == 322
    assert audio.display_offset == 4
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_owner_drain_does_not_wait_for_another_owner() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    first = PlaybackOwner(uuid4(), "agent", uuid4())
    second = PlaybackOwner(uuid4(), "answer", uuid4())
    await audio.append_owned(first, b"a" * 320, start_offset=0, end_offset=3)
    await audio.append_owned(second, b"b" * 320, start_offset=3, end_offset=7)
    first_drain = asyncio.create_task(audio.wait_until_owner_drained(first))
    second_drain = asyncio.create_task(audio.wait_until_owner_drained(second))
    await settle()

    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    await asyncio.wait_for(first_drain, timeout=0.1)
    assert not second_drain.done()
    assert audio.display_offset == 3

    process.stdout.feed_data(render_packet(audio.playback_generation, 640))
    await asyncio.wait_for(second_drain, timeout=0.1)
    assert audio.display_offset == 7
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_reset_fences_stale_ack_and_releases_old_owner_drain() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    owner = PlaybackOwner(uuid4(), "answer", uuid4())
    started: list[PlaybackOwner | None] = []
    audio.set_playback_started_callback(started.append)
    await audio.append_owned(owner, b"a" * 320, start_offset=0, end_offset=3)
    stale_generation = audio.playback_generation
    drain = asyncio.create_task(audio.wait_until_owner_drained(owner))
    await settle()

    audio.reset(8)
    await asyncio.wait_for(drain, timeout=0.1)
    process.stdout.feed_data(render_packet(stale_generation, 320))
    await settle()

    assert audio.display_offset == 8
    assert audio.rendered_playback_bytes == 0
    assert audio.played_audio_bytes == 0
    assert started == []
    assert audio._terminal_error is None

    await audio.append(b"b" * 320, start_offset=8, end_offset=12)
    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    await settle()
    assert audio.display_offset == 12
    assert audio.played_audio_bytes == 320
    assert started == [None]
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_pure_effect_counts_only_confirmed_rendered_bytes():
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=0)
    assert audio.played_audio_bytes == 0
    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    await settle()
    assert audio.played_audio_bytes == 320
    assert audio.played_offset == audio.display_offset == 0
    audio.reset()
    assert audio.played_audio_bytes == 0
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_cumulative_ack_recovers_dropped_render_packets() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    rendered: list[bytes] = []
    audio.set_rendered_audio_callback(rendered.append)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    await audio.append(b"b" * 320, start_offset=3, end_offset=7)
    await audio.append(b"c" * 320, start_offset=7, end_offset=11)
    process.stdout.feed_data(render_packet(audio.playback_generation, 960))
    await settle()

    assert audio.rendered_playback_bytes == 960
    assert audio.display_offset == 11
    assert rendered == [b"r" * 320]
    await asyncio.wait_for(audio.wait_until_drained(), timeout=0.1)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_heartbeat_can_confirm_last_ack_without_render_callback() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    started: list[PlaybackOwner | None] = []
    rendered: list[bytes] = []
    audio.set_playback_started_callback(started.append)
    audio.set_rendered_audio_callback(rendered.append)
    await audio.append(b"a" * 160, start_offset=0, end_offset=3)
    drain = asyncio.create_task(audio.wait_until_drained())
    await settle()

    process.stdout.feed_data(
        render_packet(audio.playback_generation, 160, valid_byte_count=0)
    )
    await asyncio.wait_for(drain, timeout=0.1)

    assert audio.display_offset == 3
    assert started == [None]
    assert rendered == []
    await audio.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_delta", "cumulative", "valid_count", "message"),
    [
        (1, 0, 0, "generation is ahead"),
        (0, 642, 0, "more bytes than Python queued"),
        (0, 3, 0, "PCM16 aligned"),
        (0, 320, 322, "valid byte count"),
        (0, 320, 1, "valid byte count"),
        (0, 160, 320, "exceeds its cumulative"),
    ],
)
async def test_livekit_rejects_impossible_render_ack(
    generation_delta: int,
    cumulative: int,
    valid_count: int,
    message: str,
) -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 640, start_offset=0, end_offset=3)
    process.stdout.feed_data(
        render_packet(
            audio.playback_generation + generation_delta,
            cumulative,
            valid_byte_count=valid_count,
        )
    )
    await settle()

    with pytest.raises(RuntimeError, match=message):
        await audio.run(lambda *_: None, asyncio.Event())


@pytest.mark.asyncio
@pytest.mark.parametrize("cumulative", [0, 320])
async def test_livekit_rejects_backward_or_duplicate_render_ack(
    cumulative: int,
) -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 640, start_offset=0, end_offset=3)
    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    process.stdout.feed_data(render_packet(audio.playback_generation, cumulative))
    await settle()

    with pytest.raises(RuntimeError, match="backwards|exceeds its cumulative"):
        await audio.run(lambda *_: None, asyncio.Event())


@pytest.mark.asyncio
async def test_livekit_drain_watchdog_ignores_nonprogress_heartbeats() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, drain_watchdog_timeout_seconds=0.02)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    drain = asyncio.create_task(audio.wait_until_drained())
    for _ in range(8):
        process.stdout.feed_data(
            render_packet(audio.playback_generation, 0, valid_byte_count=0)
        )
        await asyncio.sleep(0.005)

    with pytest.raises(RuntimeError, match="stopped acknowledging playback"):
        await drain
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_paused_drain_waits_until_resume_without_consuming() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, drain_watchdog_timeout_seconds=0.02)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    await audio.pause()
    drain = asyncio.create_task(audio.wait_until_drained())
    await asyncio.sleep(0.04)
    assert not drain.done()
    assert audio.display_offset == 0

    await audio.resume()
    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    await asyncio.wait_for(drain, timeout=0.1)
    assert audio.display_offset == 3
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_controls_keep_capture_alive_and_stop_fences_playback() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    generation = audio.playback_generation
    await audio.pause()
    await audio.resume()
    await audio.duck()
    await audio.unduck()
    await settle()
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    await audio.stop()
    await settle()

    controls = [
        struct.unpack("<BQ", payload)
        for magic, payload in decoded_packets(process.stdin.written)
        if magic == CONTROL_MAGIC
    ]
    assert controls == [
        (CONTROL_RESET, generation),
        (CONTROL_PAUSE, generation),
        (CONTROL_RESUME, generation),
        (CONTROL_DUCK, generation),
        (CONTROL_UNDUCK, generation),
        (CONTROL_STOP, generation),
    ]
    assert process.terminate_calls == 0
    process.stdout.feed_data(packet(1))
    await settle()
    assert audio.frame_queue_depth == 2
    await audio.wait_until_drained()
    with pytest.raises(RuntimeError, match="requires reset"):
        await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    audio.reset()
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    await audio.close()


@pytest.mark.asyncio
async def test_voice_activity_control_is_urgent_and_valid_after_playback_stop() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    generation = audio.playback_generation

    await audio.stop()
    audio.notify_voice_activity()
    await settle()

    controls = [
        struct.unpack("<BQ", payload)
        for magic, payload in decoded_packets(process.stdin.written)
        if magic == CONTROL_MAGIC
    ]
    assert controls == [
        (CONTROL_RESET, generation),
        (CONTROL_STOP, generation),
        (CONTROL_ENSURE_PROCESSED, generation),
    ]
    assert audio.playback_generation == generation
    assert audio.queued_playback_bytes == 0
    assert audio.rendered_playback_bytes == 0

    process.stdout.feed_data(packet(1))
    await settle()
    assert audio.frame_queue_depth == 2
    await audio.close()


@pytest.mark.asyncio
async def test_repeated_voice_activity_coalesces_while_advisory_is_pending() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    process.stdin.drain_gate.clear()

    audio.notify_voice_activity()
    audio.notify_voice_activity()
    await settle()
    audio.notify_voice_activity()

    controls = [
        struct.unpack("<BQ", payload)
        for magic, payload in decoded_packets(process.stdin.written)
        if magic == CONTROL_MAGIC
    ]
    assert controls == [
        (CONTROL_RESET, audio.playback_generation),
        (CONTROL_ENSURE_PROCESSED, audio.playback_generation),
    ]
    assert audio._urgent_outbound.empty()
    assert audio._ensure_processed_packet is not None

    process.stdin.drain_gate.set()
    await settle()
    assert audio._ensure_processed_packet is None
    await audio.close()


@pytest.mark.asyncio
async def test_voice_activity_drops_when_urgent_queue_is_full_without_failure() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    generation = audio.playback_generation
    process.stdin.drain_gate.clear()
    await audio.pause()
    await settle()
    for _ in range(audio._urgent_outbound.maxsize):
        audio._enqueue_control_nowait(CONTROL_PAUSE, generation)

    audio.notify_voice_activity()

    assert audio._urgent_outbound.full()
    assert audio._ensure_processed_packet is None
    assert audio._terminal_error is None
    process.stdin.drain_gate.set()
    await audio.close()


@pytest.mark.asyncio
async def test_reset_releases_pending_voice_activity_for_new_generation() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    old_generation = audio.playback_generation
    process.stdin.drain_gate.clear()
    audio.notify_voice_activity()
    await settle()
    old_packet = audio._ensure_processed_packet
    assert old_packet is not None

    audio.reset()
    new_generation = audio.playback_generation
    audio.notify_voice_activity()
    new_packet = audio._ensure_processed_packet

    assert new_generation == old_generation + 1
    assert new_packet is not None
    assert new_packet is not old_packet
    assert audio._urgent_outbound.qsize() == 2

    process.stdin.drain_gate.set()
    await settle()
    controls = [
        struct.unpack("<BQ", payload)
        for magic, payload in decoded_packets(process.stdin.written)
        if magic == CONTROL_MAGIC
    ]
    assert controls == [
        (CONTROL_RESET, old_generation),
        (CONTROL_ENSURE_PROCESSED, old_generation),
        (CONTROL_RESET, new_generation),
        (CONTROL_ENSURE_PROCESSED, new_generation),
    ]
    assert audio._ensure_processed_packet is None
    assert audio._terminal_error is None
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_capture_never_reuses_async_render_as_reverse_reference() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 640, start_offset=0, end_offset=3)
    generation = audio.playback_generation
    process.stdout.feed_data(render_packet(generation, 320, pcm=b"x" * 320))
    process.stdout.feed_data(packet(1))
    process.stdout.feed_data(render_packet(generation, 640, pcm=b"y" * 320))
    process.stdout.feed_data(packet(2))
    await settle()

    assert audio._frames.get_nowait()[1] == bytes(320)
    assert audio._frames.get_nowait()[1] == bytes(320)
    assert audio._frames.get_nowait()[1] == bytes(320)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_writer_backpressure_does_not_block_capture_or_reset() -> None:
    process = FakeProcess()
    process.stdin.drain_gate.clear()
    audio, _ = make_input(process, writer_queue_size=1)
    await start_with_first_frame(audio, process)
    old_generation = audio.playback_generation
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    blocked_append = asyncio.create_task(
        audio.append(b"b" * 320, start_offset=3, end_offset=7)
    )
    await settle()
    assert not blocked_append.done()

    process.stdout.feed_data(packet(1))
    await settle()
    assert audio.frame_queue_depth == 2
    audio.reset(10)
    assert audio.queued_playback_bytes == 0
    assert audio.display_offset == 10
    assert audio._urgent_outbound.qsize() == 1

    process.stdin.drain_gate.set()
    await asyncio.wait_for(blocked_append, timeout=0.1)
    await settle()
    sent = decoded_packets(process.stdin.written)
    assert sent[1] == (
        CONTROL_MAGIC,
        struct.pack("<BQ", CONTROL_RESET, audio.playback_generation),
    )
    assert len(sent) == 2
    process.stdout.feed_data(render_packet(old_generation, 320))
    await settle()
    assert audio.display_offset == 10
    assert audio._terminal_error is None
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_writer_error_fails_audio_run() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    process.stdin.write_error = BrokenPipeError("native input pipe closed")

    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    await settle()
    with pytest.raises(RuntimeError, match="native input pipe closed"):
        await audio.run(lambda *_: None, asyncio.Event())
    assert process.terminate_calls == 1
    assert process.stdin.closed


@pytest.mark.asyncio
async def test_livekit_close_releases_drain_and_discards_pending_packets() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    drain = asyncio.create_task(audio.wait_until_drained())
    await settle()
    await audio.close()
    await asyncio.wait_for(drain, timeout=0.1)

    assert audio._outbound.empty()
    assert process.stdin.closed
    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_livekit_render_window_prioritizes_controls_but_not_flush() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, max_inflight_playback_bytes=320)
    await start_with_first_frame(audio, process)
    generation = audio.playback_generation
    await audio.append(b"a" * 322, start_offset=0, end_offset=3)
    await settle()
    drain = asyncio.create_task(audio.wait_until_drained())
    await settle()
    await audio.pause()
    await audio.resume()
    await settle()

    sent = decoded_packets(process.stdin.written)
    assert sent == [
        (CONTROL_MAGIC, struct.pack("<BQ", CONTROL_RESET, generation)),
        (PLAYBACK_MAGIC, struct.pack("<Q", generation) + b"a" * 320),
        (CONTROL_MAGIC, struct.pack("<BQ", CONTROL_PAUSE, generation)),
        (CONTROL_MAGIC, struct.pack("<BQ", CONTROL_RESUME, generation)),
    ]
    assert not drain.done()

    process.stdout.feed_data(render_packet(generation, 320))
    await settle()
    assert decoded_packets(process.stdin.written)[-2:] == [
        (PLAYBACK_MAGIC, struct.pack("<Q", generation) + b"aa"),
        (CONTROL_MAGIC, struct.pack("<BQ", CONTROL_FLUSH, generation)),
    ]
    assert not drain.done()
    process.stdout.feed_data(render_packet(generation, 322, valid_byte_count=2))
    await asyncio.wait_for(drain, timeout=0.1)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_restart_clears_playback_and_fences_old_generation() -> None:
    processes = [FakeProcess(), FakeProcess()]
    index = 0

    async def spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        nonlocal index
        process = processes[index]
        index += 1
        return process

    audio = LiveKitProcessedPcmInput("/fake/helper", process_factory=spawn)
    await start_with_first_frame(audio, processes[0])
    await audio.append(b"a" * 640, start_offset=0, end_offset=7)
    old_generation = audio.playback_generation
    processes[0].stdout.feed_data(render_packet(old_generation, 320))
    await settle()
    assert audio.display_offset == 7
    await audio.close()

    await start_with_first_frame(audio, processes[1])
    assert audio.playback_generation > old_generation
    assert audio.display_offset == 0
    assert audio.queued_playback_bytes == 0
    assert audio.rendered_playback_bytes == 0
    processes[1].stdout.feed_data(render_packet(old_generation, 640))
    await settle()
    assert audio._terminal_error is None
    assert decoded_packets(processes[1].stdin.written) == [
        (
            CONTROL_MAGIC,
            struct.pack("<BQ", CONTROL_RESET, audio.playback_generation),
        )
    ]
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_writer_queue_timeout_fails_without_dropping_audio() -> None:
    process = FakeProcess()
    process.stdin.drain_gate.clear()
    audio, _ = make_input(
        process,
        writer_queue_size=1,
        drain_watchdog_timeout_seconds=0.01,
    )
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    old_generation = audio.playback_generation

    with pytest.raises(RuntimeError, match="stopped accepting playback"):
        await audio.append(b"b" * 320, start_offset=3, end_offset=7)

    assert audio.playback_generation > old_generation
    assert audio.queued_playback_bytes == 0
    with pytest.raises(RuntimeError, match="stopped accepting playback"):
        await audio.run(lambda *_: None, asyncio.Event())
    assert process.stdin.closed


@pytest.mark.asyncio
async def test_livekit_callback_errors_cannot_break_capture_or_drain() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)

    def failing_callback(_value: object) -> None:
        raise RuntimeError("optional visualizer failed")

    audio.set_playback_started_callback(failing_callback)
    audio.set_rendered_audio_callback(failing_callback)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    process.stdout.feed_data(render_packet(audio.playback_generation, 320))
    await settle()
    await asyncio.wait_for(audio.wait_until_drained(), timeout=0.1)

    assert audio._terminal_error is None
    assert audio.display_offset == 3
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_drain_fails_immediately_if_helper_exits() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    await start_with_first_frame(audio, process)
    await audio.append(b"a" * 320, start_offset=0, end_offset=3)
    drain = asyncio.create_task(audio.wait_until_drained())
    await settle()
    process.exit(7)

    with pytest.raises(RuntimeError, match="stdout closed unexpectedly"):
        await asyncio.wait_for(drain, timeout=0.1)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_render_heartbeat_is_not_capture_readiness() -> None:
    process = FakeProcess()
    audio, _ = make_input(process)
    start = asyncio.create_task(audio.start())
    await settle()
    process.stdout.feed_data(
        render_packet(audio.playback_generation, 0, valid_byte_count=0)
    )
    await settle()
    assert not start.done()

    process.stdout.feed_data(packet(0))
    await asyncio.wait_for(start, timeout=0.1)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_window_can_fill_a_partial_frame_from_the_next_append() -> None:
    process = FakeProcess()
    audio, _ = make_input(process, max_inflight_playback_bytes=320)
    await start_with_first_frame(audio, process)
    generation = audio.playback_generation
    await audio.append(b"aa", start_offset=0, end_offset=1)
    await audio.append(b"b" * 320, start_offset=1, end_offset=3)
    await settle()

    pcm_packets = [
        payload[8:]
        for magic, payload in decoded_packets(process.stdin.written)
        if magic == PLAYBACK_MAGIC
    ]
    assert pcm_packets == [b"aa", b"b" * 318]
    drain = asyncio.create_task(audio.wait_until_drained())
    process.stdout.feed_data(render_packet(generation, 320))
    await settle()

    assert decoded_packets(process.stdin.written)[-2:] == [
        (PLAYBACK_MAGIC, struct.pack("<Q", generation) + b"bb"),
        (CONTROL_MAGIC, struct.pack("<BQ", CONTROL_FLUSH, generation)),
    ]
    process.stdout.feed_data(render_packet(generation, 322, valid_byte_count=2))
    await asyncio.wait_for(drain, timeout=0.1)
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_capture_and_tts_start_share_one_inflight_startup() -> None:
    process = FakeProcess()
    spawn_gate = asyncio.Event()
    spawn_calls = 0

    async def spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        nonlocal spawn_calls
        spawn_calls += 1
        await spawn_gate.wait()
        return process

    audio = LiveKitProcessedPcmInput("/fake/helper", process_factory=spawn)
    first = asyncio.create_task(audio.start())
    second = asyncio.create_task(audio.start())
    await settle()

    assert spawn_calls == 1
    assert not first.done()
    assert not second.done()
    spawn_gate.set()
    await settle()
    assert not first.done()
    assert not second.done()

    process.stdout.feed_data(packet(0))
    await asyncio.wait_for(asyncio.gather(first, second), timeout=0.1)
    assert spawn_calls == 1
    assert len(process.stdin.written) == 1
    await audio.close()
    assert process.terminate_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_tts", [False, True])
async def test_owner_bound_livekit_resume_bypasses_paused_append_backpressure(
    nested_tts: bool,
) -> None:
    process = FakeProcess()
    audio, _ = make_input(
        process,
        writer_queue_size=1,
        max_inflight_playback_bytes=320,
    )
    await start_with_first_frame(audio, process)
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("answer", uuid4())
    bound = arbiter.bind_pcm(owner, audio)
    controls = (
        arbiter.bind_tts(
            owner,
            SimpleNamespace(pause=bound.pause, resume=bound.resume),
        )
        if nested_tts
        else bound
    )
    await controls.pause()
    append = asyncio.create_task(
        bound.append(b"a" * 1_600, start_offset=0, end_offset=5)
    )
    try:
        await settle()
        await settle()
        assert not append.done()
        assert arbiter._operation_lock.locked()

        await asyncio.wait_for(controls.resume(), timeout=0.1)
        await settle()
        assert (
            CONTROL_MAGIC,
            struct.pack("<BQ", CONTROL_RESUME, audio.playback_generation),
        ) in decoded_packets(process.stdin.written)

        for cumulative in range(320, 1_601, 320):
            process.stdout.feed_data(
                render_packet(audio.playback_generation, cumulative)
            )
            await settle()
            await settle()
        await asyncio.wait_for(append, timeout=0.1)
        await asyncio.wait_for(bound.wait_until_drained(), timeout=0.1)
        assert audio._terminal_error is None
        assert audio.rendered_playback_bytes == 1_600
    finally:
        append.cancel()
        await asyncio.gather(append, return_exceptions=True)
        await audio.close()
        await arbiter.close()


@pytest.mark.asyncio
async def test_livekit_ready_log_is_content_free_and_once_per_helper_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="fish_memory.companion.livekit_audio")
    processes = [FakeProcess(), FakeProcess()]
    spawn_index = 0
    times = iter([1_000_000_000, 1_123_000_000, 2_000_000_000, 2_456_000_000])

    async def spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        nonlocal spawn_index
        process = processes[spawn_index]
        spawn_index += 1
        return process

    audio = LiveKitProcessedPcmInput(
        "/fake/helper",
        process_factory=spawn,
        clock_ns=lambda: next(times),
    )
    for process in processes:
        process.stderr.feed_data(b"arbitrary-native-content-must-not-be-logged\n")
        await start_with_first_frame(audio, process)
        await audio.start()
        await audio.start()
        await audio.close()

    ready = [
        record.getMessage()
        for record in caplog.records
        if "LiveKit 本地音频已就绪" in record.getMessage()
    ]
    assert ready == [
        "LiveKit 本地音频已就绪: backend=Apple VPIO, duplex=shared, "
        "sample_rate=16000, frame_ms=10, startup_ms=123.0",
        "LiveKit 本地音频已就绪: backend=Apple VPIO, duplex=shared, "
        "sample_rate=16000, frame_ms=10, startup_ms=456.0",
    ]
    assert spawn_index == 2
    assert "arbitrary-native-content" not in caplog.text
    assert audio._startup_timeout_seconds == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "protocol", "eof", "frame_then_eof"])
async def test_livekit_ready_log_is_not_emitted_for_failed_startup(
    failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="fish_memory.companion.livekit_audio")
    process = FakeProcess()
    audio, _ = make_input(process, startup_timeout_seconds=0.01)
    start = asyncio.create_task(audio.start())
    await settle()
    if failure == "protocol":
        process.stdout.feed_data(packet(0, magic=b"BAD!"))
    elif failure == "eof":
        process.exit(7)
    elif failure == "frame_then_eof":
        process.stdout.feed_data(packet(0))
        process.exit(7)

    with pytest.raises(RuntimeError):
        await start

    assert "LiveKit 本地音频已就绪" not in caplog.text
    await audio.close()


@pytest.mark.asyncio
async def test_native_ack_advances_display_before_confirming_the_sentence_mark() -> (
    None
):
    process = FakeProcess()
    audio, _ = make_input(process)
    process.stdout.feed_data(packet(0))
    await audio.start()
    audio.reset()
    generation = audio._generation
    await audio.append(b"r" * 640, start_offset=0, end_offset=4)
    await audio.mark(4)
    assert audio.played_offset == audio.display_offset == 0
    process.stdout.feed_data(render_packet(generation, 320))
    await asyncio.sleep(0.01)
    assert audio.display_offset == 4
    assert audio.played_offset == 0
    process.stdout.feed_data(render_packet(generation, 640))
    await audio.wait_until_drained()
    assert audio.played_offset == 4
    await audio.stop()
    process.stdout.feed_data(render_packet(generation, 640, valid_byte_count=0))
    await asyncio.sleep(0.01)
    assert audio.played_offset == audio.display_offset == 0
    await audio.close()


@pytest.mark.asyncio
async def test_livekit_logs_only_known_vpio_transition_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="fish_memory.companion.livekit_audio")
    audio = LiveKitProcessedPcmInput("/fake/helper")
    stderr = asyncio.StreamReader()
    stderr.feed_data(
        b"fish-memory-audio state adaptive-vpio transition uptimeNs=1 "
        b"captureSequence=3989 reason=quiet-standby requestedBypassed=true "
        b"actualBypassed=true playbackGeneration=none succeeded=true\n"
        b"voiceOutput=Private Device Name arbitrary-native-content\n"
        b"fish-memory-audio state adaptive-vpio transition reason=playback-prewarm "
    )
    stderr.feed_data(b"actualBypassed=false playbackGeneration=3 succeeded=true\n")
    stderr.feed_eof()
    await audio._drain_stderr(stderr)

    logged = [r.getMessage() for r in caplog.records if "VPIO" in r.getMessage()]
    assert logged == [
        "VPIO 省电切换: reason=quiet-standby actualBypassed=true "
        "playbackGeneration=none succeeded=true",
        "VPIO 省电切换: reason=playback-prewarm actualBypassed=false "
        "playbackGeneration=3 succeeded=true",
    ]
    assert "Private Device" not in caplog.text
    assert "arbitrary-native-content" not in caplog.text
