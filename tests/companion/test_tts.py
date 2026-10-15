import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosedError

import fish_memory.companion.tts as tts_module
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.tts import ApiTtsOutput, TtsConfiguration


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(self, name: str, **kwargs: object) -> None:
        self.events.append((name, kwargs))


class FakeSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self._played_offset = 0

    @property
    def played_offset(self) -> int:
        return self._played_offset

    @property
    def display_offset(self) -> int:
        return self._played_offset

    async def mark(self, end_offset: int) -> None:
        self._played_offset = max(self._played_offset, end_offset)

    async def start(self) -> None:
        self.calls.append(("start", None))

    def reset(self, text_offset: int = 0) -> None:
        self.calls.append(("reset", text_offset))
        self._played_offset = text_offset

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        self.calls.append(("append", (pcm, start_offset, end_offset)))
        self._played_offset = max(self._played_offset, end_offset)

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))

    async def wait_until_drained(self) -> None:
        return None


class DelayedDrainSink(FakeSink):
    def __init__(self) -> None:
        super().__init__()
        self.drain_started = asyncio.Event()
        self.drained = asyncio.Event()

    async def wait_until_drained(self) -> None:
        self.drain_started.set()
        await self.drained.wait()


class FakeSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send(self, value: str) -> None:
        message = json.loads(value)
        self.sent.append(message)
        action = message["type"]
        if action == "start":
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "ready",
                        "provider": "local",
                        "model": "matcha-icefall-zh-en",
                        "voice": "default",
                        "sampleRate": message["sample_rate"],
                    }
                )
            )
        elif action == "append":
            await self.incoming.put(
                json.dumps({"type": "audio", "text": message["text"]})
            )
            await self.incoming.put(b"\x01\x02")
        elif action in {"finish", "stop"}:
            await self.incoming.put(json.dumps({"type": "finished"}))

    async def recv(self) -> str | bytes:
        value = await self.incoming.get()
        if value is None:
            raise RuntimeError("closed")
        return value

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)


class MultiPacketSentenceSocket(FakeSocket):
    """Emit two PCM frames for every provider sentence."""

    def __init__(self) -> None:
        super().__init__()
        self._sentence_index = 0

    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] != "append":
            await super().send(value)
            return

        self.sent.append(message)
        sentence_index = self._sentence_index
        self._sentence_index += 1
        await self.incoming.put(
            json.dumps(
                {
                    "type": "sentence_begin",
                    "text": message["text"],
                    "sentenceIndex": sentence_index,
                }
            )
        )
        for pcm in (b"\x01\x02", b"\x03\x04"):
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "audio",
                        "sentenceIndex": sentence_index,
                    }
                )
            )
            await self.incoming.put(pcm)


class ReusedSentenceIndexSocket(FakeSocket):
    """Reuse one provider sentence index across distinct append requests."""

    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] != "append":
            await super().send(value)
            return

        self.sent.append(message)
        for pcm in (b"\x01\x02", b"\x03\x04"):
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "audio",
                        "text": message["text"],
                        "sentenceIndex": 0,
                    }
                )
            )
            await self.incoming.put(pcm)


class MergedAppendSentenceSocket(FakeSocket):
    """Merge later append requests into one announced provider sentence."""

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []

    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "append":
            self.sent.append(message)
            self._texts.append(message["text"])
            return
        if message["type"] != "finish":
            await super().send(value)
            return

        self.sent.append(message)
        provider_sentences = (self._texts[0], "".join(self._texts[1:]))
        for sentence_index, text in enumerate(provider_sentences):
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "sentence_begin",
                        "text": text,
                        "sentenceIndex": sentence_index,
                    }
                )
            )
            for pcm in (b"\x01\x02", b"\x03\x04"):
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "audio",
                            "text": "",
                            "sentenceIndex": sentence_index,
                        }
                    )
                )
                await self.incoming.put(pcm)
        await self.incoming.put(json.dumps({"type": "finished"}))


@pytest.mark.asyncio
async def test_default_tts_connector_sends_api_key_as_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def connect(url: str, **kwargs: object) -> FakeSocket:
        captured["url"] = url
        captured.update(kwargs)
        return FakeSocket()

    monkeypatch.setattr(tts_module.websockets, "connect", connect)
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        api_key="fm_companion_key",
    )

    socket = await output._connect("ws://127.0.0.1:8000/v1/tts/stream")
    await socket.close()

    assert captured["additional_headers"] == {
        "Authorization": "Bearer fm_companion_key"
    }


class UnannotatedAudioSocket(FakeSocket):
    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "append":
            self.sent.append(message)
            await self.incoming.put(json.dumps({"type": "audio"}))
            await self.incoming.put(b"\x01\x02")
            return
        await super().send(value)


class DelayedStopSocket(FakeSocket):
    def __init__(self) -> None:
        super().__init__()
        self.stop_received = asyncio.Event()

    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] != "stop":
            await super().send(value)
            return
        self.sent.append(message)
        self.stop_received.set()


class CancellationResistantReceiveSocket(DelayedStopSocket):
    """Keep one recv alive after cancellation until a test-controlled release."""

    def __init__(self) -> None:
        super().__init__()
        self.receive_blocked = asyncio.Event()
        self.receive_cancelled = asyncio.Event()
        self.release_receive = asyncio.Event()
        self._receive_count = 0

    async def recv(self) -> str | bytes:
        self._receive_count += 1
        if self._receive_count == 1:
            return await super().recv()

        self.receive_blocked.set()
        try:
            await self.release_receive.wait()
        except asyncio.CancelledError:
            # Model a native/network recv that reports cancellation only after
            # it has already obtained one more frame.
            self.receive_cancelled.set()
            await self.release_receive.wait()
        return await super().recv()


class CloseOnFirstAppendSocket(FakeSocket):
    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "append":
            self.closed = True
            raise ConnectionClosedError(None, None)
        await super().send(value)


class FinishesImmediatelySocket(FakeSocket):
    async def send(self, value: str) -> None:
        message = json.loads(value)
        await super().send(value)
        if message["type"] == "start":
            await self.incoming.put(json.dumps({"type": "finished"}))


class ManualAudioSocket(FakeSocket):
    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "append":
            self.sent.append(message)
            return
        await super().send(value)


@pytest.mark.asyncio
async def test_tts_delegates_candidate_ducking_to_pcm_sink() -> None:
    sink = FakeSink()
    output = ApiTtsOutput("http://127.0.0.1:8000", sink=sink)

    await output.duck()
    await output.unduck()

    assert sink.calls == [("duck", None), ("unduck", None)]


@pytest.mark.asyncio
async def test_tts_reuses_api_websocket_between_answer_turns() -> None:
    socket = FakeSocket()
    connects = 0

    async def connector(_: str) -> FakeSocket:
        nonlocal connects
        connects += 1
        return socket

    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=connector,
    )

    await output.start()
    await output.append("你好")
    await asyncio.sleep(0)
    await output.finish()
    await output.start()
    await output.append("再见")
    await asyncio.sleep(0)
    await output.stop()
    await output.close()

    assert connects == 1
    assert [message["type"] for message in socket.sent] == [
        "start",
        "append",
        "finish",
        "start",
        "append",
        "stop",
    ]
    assert ("append", (b"\x01\x02", 0, 2)) in sink.calls


@pytest.mark.asyncio
async def test_tts_refreshes_configuration_before_each_spoken_turn() -> None:
    socket = FakeSocket()
    refresh_count = 0

    async def refresh_configuration() -> None:
        nonlocal refresh_count
        refresh_count += 1

    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=lambda _: asyncio.sleep(0, result=socket),
        before_start=refresh_configuration,
    )

    await output.start()
    await output.finish()
    await output.start()
    await output.stop()
    await output.close()

    assert refresh_count == 2


@pytest.mark.asyncio
async def test_tts_uses_configured_fish_voice_for_next_connection() -> None:
    socket = FakeSocket()

    async def connector(_: str) -> FakeSocket:
        return socket

    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=connector,
    )
    output.configure(
        model="qwen-audio-3.0-tts-flash",
        voice="qwen-audio-3.0-tts-flash-mine-1",
    )

    await output.start()
    await output.stop()
    await output.close()

    assert socket.sent[0] == {
        "type": "start",
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "qwen-audio-3.0-tts-flash-mine-1",
        "audio_format": "pcm",
        "sample_rate": 22050,
    }


def test_tts_exposes_exact_provider_configuration_for_promotion() -> None:
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        sample_rate=16_000,
    )

    output.configure(model="model-new", voice="voice-new")

    assert output.configuration == TtsConfiguration(
        model="model-new",
        voice="voice-new",
        sample_rate=16_000,
    )


@pytest.mark.asyncio
async def test_tts_exposes_actual_ready_provider_state() -> None:
    socket = FakeSocket()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        sample_rate=16_000,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()

    state = output.provider_state
    assert state.ready is True
    assert state.active is True
    assert state.error_free is True
    assert state.provider == "local"
    assert state.configuration == TtsConfiguration(
        "matcha-icefall-zh-en",
        "default",
        16_000,
    )
    await output.close()


@pytest.mark.asyncio
async def test_semantic_flush_preserves_connection_and_duplicate_text_offsets() -> None:
    class BoundarySocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.pending = ""

        async def send(self, value):
            message = json.loads(value)
            if message["type"] == "append":
                self.sent.append(message)
                self.pending += message["text"]
            elif message["type"] == "flush":
                self.sent.append(message)
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "audio",
                            "text": self.pending,
                            "sentenceIndex": 0,
                        }
                    )
                )
                await self.incoming.put(b"pcm")
                await self.incoming.put(json.dumps({"type": "segment_finished"}))
                self.pending = ""
            else:
                await super().send(value)

    socket, sink = BoundarySocket(), FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )
    try:
        await output.start()
        await output.flush()  # No text: do not rotate an empty task.
        for text in ("我去查查", "我去查查"):
            await output.append(text)
            await output.flush()
            await output.flush()  # Duplicate completion notification.
            for _ in range(10):
                await asyncio.sleep(0)
        assert output.provider_state.active is True
        await output.finish()
        assert [m["type"] for m in socket.sent] == [
            "start",
            "append",
            "flush",
            "append",
            "flush",
            "finish",
        ]
        ranges = [call[1][1:] for call in sink.calls if call[0] == "append"]
        assert ranges == [(0, 4), (4, 8)]
        assert output.played_offset == 8
    finally:
        await output.close()


@pytest.mark.asyncio
async def test_adopted_provider_records_public_text_and_audio_timings():
    socket = FakeSocket()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=lambda _: _return_socket(socket),
    )
    diagnostics = RecordingDiagnostics()
    try:
        await output.start()
        assert diagnostics.events == []
        output.activate_diagnostics(diagnostics)
        output.activate_diagnostics(diagnostics)
        await output.append("首段文字")
        await output.flush()
        await output.finish()
        names = [name for name, _ in diagnostics.events]
        assert names.count("tts_ready") == 1
        assert names.count("tts_text_submitted") == 1
        assert names.count("tts_first_audio_received") == 1
        assert names.count("tts_first_pcm_queued") == 1
        assert [m["type"] for m in socket.sent].count("start") == 1
    finally:
        await output.close()


@pytest.mark.asyncio
async def test_tts_reports_transport_boundaries_and_connection_reuse() -> None:
    socket = FakeSocket()
    diagnostics = RecordingDiagnostics()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=lambda _: _return_socket(socket),
        diagnostics=diagnostics,
    )

    await output.start()
    await output.append("你好")
    await asyncio.sleep(0)
    await output.finish()
    await output.start()

    names = [name for name, _ in diagnostics.events]
    assert names.count("tts_ready") == 2
    assert names.count("tts_text_submitted") == 1
    assert names.count("tts_first_audio_received") == 1
    assert names.count("tts_first_pcm_queued") == 1
    ready_events = [event for event in diagnostics.events if event[0] == "tts_ready"]
    assert ready_events[0][1]["metadata"] == {
        "connectionReused": False,
        "provider": "local",
        "model": "matcha-icefall-zh-en",
        "voice": "default",
    }
    assert ready_events[1][1]["metadata"] == {
        "connectionReused": True,
        "provider": "local",
        "model": "matcha-icefall-zh-en",
        "voice": "default",
    }

    await output.close()


@pytest.mark.asyncio
async def test_tts_does_not_guess_when_audio_has_no_text_annotation() -> None:
    socket = UnannotatedAudioSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    await output.append("一段回答")
    await asyncio.sleep(0)

    assert ("append", (b"\x01\x02", 0, 0)) in sink.calls


@pytest.mark.asyncio
async def test_tts_maps_audio_to_absolute_answer_ranges() -> None:
    socket = FakeSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    await output.append("前缀")
    await asyncio.sleep(0)
    await output.append("后面的回答")
    await asyncio.sleep(0)

    assert ("append", (b"\x01\x02", 0, 2)) in sink.calls
    assert ("append", (b"\x01\x02", 2, 7)) in sink.calls

    await output.close()


@pytest.mark.asyncio
async def test_tts_resume_offset_is_absolute_before_and_after_first_pcm() -> None:
    socket = FakeSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start(text_offset=7)
    assert output.played_offset == 7
    await output.append("后续")
    await asyncio.sleep(0)

    assert ("append", (b"\x01\x02", 7, 9)) in sink.calls
    assert output.played_offset == 9
    await output.close()


@pytest.mark.asyncio
async def test_tts_receive_loop_drops_pcm_after_playback_owner_is_stale() -> None:
    socket = ManualAudioSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("answer", None)
    bound = arbiter.bind_tts(owner, output)

    await bound.start()
    await bound.append("旧回答")
    sink.calls.clear()
    assert await arbiter.release(owner) is True
    replacement = await arbiter.acquire("agent", None)
    await socket.incoming.put(json.dumps({"type": "audio", "text": "旧回答"}))
    await socket.incoming.put(b"\x01\x02")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not any(call[0] == "append" for call in sink.calls)
    assert arbiter.is_current(replacement)
    await output.close()


@pytest.mark.asyncio
async def test_old_receive_loop_cannot_borrow_replacement_owner_token() -> None:
    sockets = [ManualAudioSocket(), ManualAudioSocket()]
    connect_count = 0

    async def connector(_: str) -> ManualAudioSocket:
        nonlocal connect_count
        socket = sockets[connect_count]
        connect_count += 1
        return socket

    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=connector,
    )
    arbiter = PlaybackArbiter()
    old_owner = await arbiter.acquire("answer", None)
    old = arbiter.bind_tts(old_owner, output)
    await old.start()
    await old.append("旧回答")
    assert await arbiter.release(old_owner) is True

    new_owner = await arbiter.acquire("agent", None)
    new = arbiter.bind_tts(new_owner, output)
    await new.start(text_offset=5)
    sink.calls.clear()
    await sockets[0].incoming.put(json.dumps({"type": "audio", "text": "旧回答"}))
    await sockets[0].incoming.put(b"\x01\x02")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert connect_count == 2
    assert not any(call[0] == "append" for call in sink.calls)
    await output.close()


@pytest.mark.asyncio
async def test_real_bound_tts_start_audio_finish_does_not_reenter_pcm_lock() -> None:
    socket = FakeSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("answer", None)
    bound = arbiter.bind_tts(owner, output)

    await asyncio.wait_for(bound.start(), timeout=0.1)
    await bound.append("真实回答")
    await asyncio.sleep(0)
    await asyncio.wait_for(bound.finish(), timeout=0.1)

    assert ("append", (b"\x01\x02", 0, 4)) in sink.calls
    assert await arbiter.release(owner) is True
    await output.close()


@pytest.mark.asyncio
async def test_tts_keeps_multiple_audio_packets_in_one_sentence_on_one_text_range() -> (
    None
):
    socket = MultiPacketSentenceSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    await output.append("第一段")
    await asyncio.sleep(0)
    await output.append("第二段")
    await asyncio.sleep(0)

    assert [call[1] for call in sink.calls if call[0] == "append"] == [
        (b"\x01\x02", 0, 3),
        (b"\x03\x04", 0, 3),
        (b"\x01\x02", 3, 6),
        (b"\x03\x04", 3, 6),
    ]

    await output.close()


@pytest.mark.asyncio
async def test_tts_reused_sentence_index_still_advances_for_new_append_text() -> None:
    socket = ReusedSentenceIndexSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    await output.append("第一段")
    await asyncio.sleep(0)
    await output.append("第二段")
    await asyncio.sleep(0)

    assert [call[1] for call in sink.calls if call[0] == "append"] == [
        (b"\x01\x02", 0, 3),
        (b"\x03\x04", 0, 3),
        (b"\x01\x02", 3, 6),
        (b"\x03\x04", 3, 6),
    ]

    await output.close()


@pytest.mark.asyncio
async def test_tts_maps_provider_sentence_merged_across_append_segments() -> None:
    socket = MergedAppendSentenceSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    for text in ("第一段", "第二段", "第三段"):
        await output.append(text)
    await output.finish()

    assert [call[1] for call in sink.calls if call[0] == "append"] == [
        (b"\x01\x02", 0, 3),
        (b"\x03\x04", 0, 3),
        (b"\x01\x02", 3, 9),
        (b"\x03\x04", 3, 9),
    ]

    await output.close()


@pytest.mark.asyncio
async def test_tts_reconnects_when_reused_websocket_is_closed() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    connects = 0

    async def connector(_: str) -> FakeSocket:
        nonlocal connects
        socket = sockets[connects]
        connects += 1
        return socket

    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=connector,
    )

    await output.start()
    await output.append("第一轮")
    await asyncio.sleep(0)
    await output.finish()

    sockets[0].closed = True
    await output.start()
    await output.append("第二轮")
    await asyncio.sleep(0)
    await output.stop()
    await output.close()

    assert connects == 2
    assert [message["type"] for message in sockets[1].sent] == [
        "start",
        "append",
        "stop",
    ]


@pytest.mark.asyncio
async def test_tts_reconnects_when_socket_closes_before_first_append() -> None:
    sockets = [CloseOnFirstAppendSocket(), FakeSocket()]
    connects = 0

    async def connector(_: str) -> FakeSocket:
        nonlocal connects
        socket = sockets[connects]
        connects += 1
        return socket

    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=connector,
    )

    await output.start()
    await output.append("恢复播放")
    await asyncio.sleep(0)
    await output.stop()
    await output.close()

    assert connects == 2
    assert [message["type"] for message in sockets[1].sent] == [
        "start",
        "append",
        "stop",
    ]


@pytest.mark.asyncio
async def test_tts_starts_a_new_provider_task_after_server_finishes_task() -> None:
    socket = FinishesImmediatelySocket()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    for _ in range(10):
        if not output._provider_task_active:
            break
        await asyncio.sleep(0)
    await output.append("下一轮回答")
    await asyncio.sleep(0)

    assert [message["type"] for message in socket.sent] == [
        "start",
        "start",
        "append",
    ]

    await output.close()


@pytest.mark.asyncio
async def test_tts_interrupt_stops_local_playback_before_provider_acknowledges() -> (
    None
):
    socket = DelayedStopSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    sink.calls.clear()
    stop_task = asyncio.create_task(output.stop())
    await asyncio.wait_for(socket.stop_received.wait(), timeout=0.1)

    assert sink.calls == [("reset", 0)]

    await socket.incoming.put(json.dumps({"type": "finished"}))
    await stop_task
    await output.close()


@pytest.mark.asyncio
async def test_tts_interrupt_starts_next_task_immediately() -> None:
    sockets = [DelayedStopSocket(), FakeSocket()]
    connects = 0

    async def connector(_: str) -> FakeSocket:
        nonlocal connects
        socket = sockets[connects]
        connects += 1
        return socket

    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=FakeSink(),
        websocket_connector=connector,
    )

    await output.start()
    await output.interrupt()
    await output.start()

    assert connects == 2
    assert [message["type"] for message in sockets[1].sent] == ["start"]

    await output.close()


@pytest.mark.asyncio
async def test_tts_interrupt_discards_late_audio_from_cancelled_task() -> None:
    socket = DelayedStopSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    sink.calls.clear()
    stop_task = asyncio.create_task(output.stop())
    await asyncio.wait_for(socket.stop_received.wait(), timeout=0.1)
    await socket.incoming.put(json.dumps({"type": "audio", "text": "旧回答"}))
    await socket.incoming.put(b"\x01\x02")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not any(call[0] == "append" for call in sink.calls)

    await socket.incoming.put(json.dumps({"type": "finished"}))
    await stop_task
    await output.close()


@pytest.mark.asyncio
async def test_tts_detached_connection_cannot_mutate_new_configured_turn() -> None:
    old_socket = CancellationResistantReceiveSocket()
    new_socket = FakeSocket()
    sockets = [old_socket, new_socket]
    connects = 0

    async def connector(_: str) -> FakeSocket:
        nonlocal connects
        socket = sockets[connects]
        connects += 1
        return socket

    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=connector,
    )

    await output.start()
    old_receive_task = output._receive_task
    assert old_receive_task is not None
    await asyncio.wait_for(old_socket.receive_blocked.wait(), timeout=0.1)
    # Queue frames that would corrupt the next task if the detached receive
    # loop retained access to ApiTtsOutput's shared events and PCM sink.
    await old_socket.incoming.put(
        json.dumps(
            {
                "type": "ready",
                "provider": "stale-provider",
                "model": "stale-model",
                "voice": "stale-voice",
            }
        )
    )
    await old_socket.incoming.put(json.dumps({"type": "audio", "text": "旧回答"}))
    await old_socket.incoming.put(b"old-pcm")
    await old_socket.incoming.put(
        json.dumps({"type": "error", "message": "stale failure"})
    )
    await old_socket.incoming.put(json.dumps({"type": "finished"}))

    await output.interrupt()
    await asyncio.wait_for(old_socket.receive_cancelled.wait(), timeout=0.1)
    output.configure(
        model="qwen-audio-3.0-tts-flash",
        voice="qwen-audio-3.0-tts-flash-mine-new",
    )
    await output.start()
    await output.append("新回答")
    await asyncio.sleep(0)

    # Let the cancellation-resistant old recv return only after the newly
    # configured provider task is fully active.
    old_socket.release_receive.set()
    await asyncio.wait_for(asyncio.shield(old_receive_task), timeout=0.1)

    appended = [call[1] for call in sink.calls if call[0] == "append"]
    assert appended == [(b"\x01\x02", 0, 3)]
    assert output._actual_provider == "local"
    assert output._actual_model == "matcha-icefall-zh-en"
    assert output._actual_voice == "default"
    assert output._error is None
    assert output._provider_task_active is True
    assert new_socket.sent[0] == {
        "type": "start",
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "qwen-audio-3.0-tts-flash-mine-new",
        "audio_format": "pcm",
        "sample_rate": 22050,
    }

    await output.finish()
    await output.close()


@pytest.mark.asyncio
async def test_tts_close_is_bounded_when_detached_recv_ignores_cancellation() -> None:
    socket = CancellationResistantReceiveSocket()
    sink = FakeSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )
    loop = asyncio.get_running_loop()
    unhandled_contexts: list[dict[str, object]] = []
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: unhandled_contexts.append(context)
    )

    try:
        await output.start()
        receive_task = output._receive_task
        assert receive_task is not None
        await asyncio.wait_for(socket.receive_blocked.wait(), timeout=0.1)

        await output.interrupt()
        await asyncio.wait_for(socket.receive_cancelled.wait(), timeout=0.1)
        started_at = loop.time()
        await asyncio.wait_for(output.close(), timeout=0.5)

        assert loop.time() - started_at < 0.5
        assert socket.closed is True
        assert receive_task.done() is False
        assert unhandled_contexts == []

        # The test never releases recv during cleanup. Release it only now so
        # the test itself leaves no background task behind; its late transport
        # error must be consumed rather than reported by the event loop.
        socket.release_receive.set()
        await asyncio.wait_for(asyncio.shield(receive_task), timeout=0.1)
        await asyncio.sleep(0)
        assert unhandled_contexts == []
    finally:
        socket.release_receive.set()
        loop.set_exception_handler(previous_exception_handler)


@pytest.mark.asyncio
async def test_tts_finish_waits_for_local_playback_to_drain() -> None:
    socket = FakeSocket()
    sink = DelayedDrainSink()
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )

    await output.start()
    await output.append("你好")
    await asyncio.sleep(0)
    finish_task = asyncio.create_task(output.finish())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(finish_task), timeout=0.01)
    sink.drained.set()
    await finish_task
    await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_physical_call"),
    [
        ("pause", ("pause", None)),
        ("interrupt", ("reset", 0)),
    ],
)
async def test_owner_bound_tts_control_preempts_blocked_physical_drain(
    operation: str,
    expected_physical_call: tuple[str, object | None],
) -> None:
    socket = FakeSocket()
    sink = DelayedDrainSink()
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire("agent", None)
    output = ApiTtsOutput(
        "http://127.0.0.1:8000",
        sink=sink,
        websocket_connector=lambda _: _return_socket(socket),
    )
    bound = arbiter.bind_tts(owner, output)
    finish_task: asyncio.Task[None] | None = None

    try:
        await bound.start()
        await bound.append("这是一段仍在物理播放队列中的回答。")
        finish_task = asyncio.create_task(bound.finish())
        await asyncio.wait_for(sink.drain_started.wait(), timeout=0.1)
        control_calls_before = sink.calls.count(expected_physical_call)

        control = bound.pause if operation == "pause" else bound.interrupt
        await asyncio.wait_for(control(), timeout=0.1)

        assert sink.calls.count(expected_physical_call) == control_calls_before + 1
        assert sink.drained.is_set() is False
        assert finish_task.done() is False
    finally:
        sink.drained.set()
        if finish_task is not None:
            await asyncio.gather(finish_task, return_exceptions=True)
        await output.close()
        await arbiter.release(owner)


async def _return_socket(socket: FakeSocket) -> FakeSocket:
    return socket


class FacialSocket(FakeSocket):
    def __init__(self):
        super().__init__()
        self.stream_number = 0
        self.text_offset = 0

    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "start":
            self.sent.append(message)
            self.stream_number += 1
            self.text_offset = 0
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "ready",
                        "facialStreamId": f"stream-{self.stream_number}",
                        "sampleRate": message["sample_rate"],
                    }
                )
            )
            return
        if message["type"] != "append":
            await super().send(value)
            return
        self.sent.append(message)
        text = message["text"]
        start = self.text_offset
        self.text_offset += len(text)
        for item in (
            {"type": "sentence_begin", "sentenceIndex": 0, "text": text},
            {
                "type": "facial",
                "sentenceIndex": 0,
                "text": text,
                "profileId": "model-v1",
                "streamId": f"stream-{self.stream_number}",
                "startOffset": start,
                "endOffset": self.text_offset,
                "targets": {"ParamAngleX": 12},
                "waitMs": 0,
            },
            {"type": "audio", "sentenceIndex": 0, "text": ""},
        ):
            await self.incoming.put(json.dumps(item))
        await self.incoming.put(b"\x00\x00")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_wait", [0, 750])
async def test_first_packet_timings_survive_native_receive_without_later_overwrite(
    first_wait,
):
    class TimedSocket(FakeSocket):
        async def send(self, value):
            message = json.loads(value)
            if message["type"] != "append":
                await super().send(value)
                return
            for index, wait in [(0, first_wait), (1, 900)]:
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "audio",
                            "text": message["text"],
                            "sentenceIndex": index,
                            "firstPacketMs": 400 if index == 0 else None,
                            "facialWaitMs": wait,
                        }
                    )
                )
                await self.incoming.put(b"\x00\x00")

    socket = TimedSocket()
    diagnostics = RecordingDiagnostics()
    output = ApiTtsOutput(
        "http://localhost",
        sink=FakeSink(),
        diagnostics=diagnostics,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    try:
        await output.start()
        await output.append("测试。")
        await output.finish()
        timing = [
            data["metadata"]
            for name, data in diagnostics.events
            if name == "tts_first_audio_timing"
        ]
        assert timing == [{"providerFirstPacketMs": 400, "facialWaitMs": first_wait}]
    finally:
        await output.close()


class DeviceClockSink(FakeSink):
    _display_offset = 0

    @property
    def display_offset(self):
        return self._display_offset

    def reset(self, text_offset=0):
        super().reset(text_offset)
        self._display_offset = text_offset

    async def mark(self, end_offset):
        self.calls.append(("mark", end_offset))

    async def append(self, pcm, *, start_offset, end_offset):
        # Receiving/queuing PCM is deliberately NOT device consumption.
        self.calls.append(("append", (pcm, start_offset, end_offset)))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["answer", "proactive", "agent"])
async def test_facial_cue_follows_device_progress_for_all_public_owners(kind):
    socket, sink = FacialSocket(), DeviceClockSink()

    async def connect(_):
        return socket

    output = ApiTtsOutput(
        "http://localhost", sink=sink, facial=True, websocket_connector=connect
    )
    arbiter = PlaybackArbiter()
    owner = await arbiter.acquire(kind, None)
    bound = arbiter.bind_tts(owner, output)
    try:
        await bound.start()
        await bound.append("你好。")
        for _ in range(8):
            await asyncio.sleep(0)
        assert socket.sent[0]["facial"] is True
        assert bound.pop_facial_cue() is None
        sink._display_offset = len("你好。")
        assert bound.pop_facial_cue() == {
            "profileId": "model-v1",
            "targets": {"ParamAngleX": 12},
        }
        assert bound.pop_facial_cue() is None
        await bound.append("再见。")
        for _ in range(8):
            await asyncio.sleep(0)
        await arbiter.release(owner)
        await arbiter.acquire("answer", None)
        sink._played_offset = 99
        assert bound.pop_facial_cue() is None
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_draft_facial_cue_remains_private_then_is_adopted_without_resynthesis():
    from fish_memory.companion.draft_pcm import PromotablePcmSink

    socket, live = FacialSocket(), DeviceClockSink()
    arbiter = PlaybackArbiter()
    private = PromotablePcmSink(live, 1000, playback_arbiter=arbiter)

    async def connect(_):
        return socket

    output = ApiTtsOutput(
        "http://localhost", sink=private, facial=True, websocket_connector=connect
    )
    try:
        await output.start()
        await output.append("你好。")
        for _ in range(8):
            await asyncio.sleep(0)
        assert private.buffered_bytes > 0
        assert live.calls == []
        assert output.pop_facial_cue() is None
        before = list(socket.sent)
        owner = await arbiter.acquire("answer", None)
        assert await private.promote(expected_end_offset=len("你好。"), owner=owner)
        bound = arbiter.bind_tts(owner, output)
        assert bound.pop_facial_cue() is None
        live._display_offset = len("你好。")
        assert bound.pop_facial_cue()["targets"] == {"ParamAngleX": 12}
        assert socket.sent == before
        await output.interrupt()
        assert output.pop_facial_cue() is None
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_adoption_binds_face_before_even_the_first_immediate_device_frame():
    from fish_memory.companion.draft_pcm import PromotablePcmSink

    socket, live = FacialSocket(), DeviceClockSink()
    arbiter = PlaybackArbiter()
    private = PromotablePcmSink(live, 1000, playback_arbiter=arbiter)

    async def connect(_):
        return socket

    output = ApiTtsOutput(
        "http://localhost", sink=private, facial=True, websocket_connector=connect
    )
    rendered = []

    async def consume_immediately(pcm, *, start_offset, end_offset):
        live._display_offset = end_offset
        rendered.append(arbiter.current_tts.pop_facial_cue())

    live.append = consume_immediately
    try:
        await output.start()
        await output.append("你好。")
        for _ in range(8):
            await asyncio.sleep(0)
        assert rendered == []
        owner = await arbiter.acquire("answer", None)
        assert await private.promote(
            expected_end_offset=3,
            owner=owner,
            before_playback=lambda: arbiter.bind_tts(owner, output),
        )
        assert rendered == [{"profileId": "model-v1", "targets": {"ParamAngleX": 12}}]
    finally:
        await output.close_transport()


class LateFacialSocket(FacialSocket):
    async def send(self, value):
        message = json.loads(value)
        if message["type"] == "append":
            self.sent.append(message)
            self.text_offset += len(message["text"])
            return
        await super().send(value)

    async def audio(self, text, index=0):
        for item in (
            {"type": "sentence_begin", "sentenceIndex": index, "text": text},
            {"type": "audio", "sentenceIndex": index, "text": ""},
        ):
            await self.incoming.put(json.dumps(item))
        await self.incoming.put(b"\x00\x00" * 160)
        await self.incoming.put(
            json.dumps({"type": "sentence_end", "sentenceIndex": index})
        )

    async def face(self, start, end, text, *, stream=None, angle=12):
        await self.incoming.put(
            json.dumps(
                {
                    "type": "facial",
                    "profileId": "model-v1",
                    "streamId": stream or f"stream-{self.stream_number}",
                    "startOffset": start,
                    "endOffset": end,
                    "text": text,
                    "targets": {"ParamAngleX": angle},
                    "waitMs": 0,
                }
            )
        )


async def drain_native():
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_late_faces_use_explicit_offsets_and_never_rewind_audio_mapping():
    socket, sink = LateFacialSocket(), DeviceClockSink()
    output = ApiTtsOutput(
        "http://localhost",
        sink=sink,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    try:
        await output.start(text_offset=20)
        await output.append("甲。乙。")
        await socket.audio("甲。", 0)
        await socket.audio("乙。", 1)
        await drain_native()
        assert output._audio_search_offset == 24
        assert output._active_audio_range == (22, 24)
        sink._display_offset = 22
        # Future result arrives first. First sentence arrives only after the
        # API has mapped all later audio; it must never advance that mapping.
        await socket.face(2, 4, "乙。", angle=20)
        await socket.face(0, 2, "甲。", angle=10)
        await drain_native()
        assert output._audio_search_offset == 24
        assert output._active_audio_range == (22, 24)
        assert output.pop_facial_cue()["targets"] == {"ParamAngleX": 10}
        assert output.pop_facial_cue() is None
        sink._played_offset = 22
        sink._display_offset = 24
        assert output.pop_facial_cue()["targets"] == {"ParamAngleX": 20}
        await socket.face(0, 2, "甲。", angle=-10)
        await drain_native()
        assert output.pop_facial_cue() is None  # Past sentence, not just late.
        sink._played_offset = 24
        await socket.face(2, 4, "乙。", angle=-20)
        await drain_native()
        assert output.pop_facial_cue() is None  # Entire utterance has finished.
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_late_face_after_provider_finished_is_used_while_device_drains():
    class DrainingSink(DeviceClockSink):
        def __init__(self):
            super().__init__()
            self.drain_started = asyncio.Event()
            self.drained = asyncio.Event()

        async def wait_until_drained(self):
            self.drain_started.set()
            await self.drained.wait()

    socket, sink = LateFacialSocket(), DrainingSink()
    output = ApiTtsOutput(
        "http://localhost",
        sink=sink,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    finish = None
    try:
        await output.start()
        await output.append("你好。")
        await socket.audio("你好。")
        finish = asyncio.create_task(output.finish())
        await asyncio.wait_for(sink.drain_started.wait(), 0.5)
        assert output._finished.is_set() and not finish.done()
        sink._display_offset = 3
        await socket.face(0, 3, "你好。")
        await drain_native()
        assert output.pop_facial_cue()["targets"] == {"ParamAngleX": 12}
        sink._played_offset = 3
        sink.drained.set()
        await finish
    finally:
        sink.drained.set()
        if finish:
            await asyncio.gather(finish, return_exceptions=True)
        await output.close_transport()


@pytest.mark.asyncio
async def test_old_stream_same_text_and_offsets_cannot_contaminate_reused_socket():
    socket, sink = LateFacialSocket(), DeviceClockSink()
    output = ApiTtsOutput(
        "http://localhost",
        sink=sink,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    try:
        await output.start()
        await output.append("你好。")
        await socket.audio("你好。")
        await output.finish()
        old_stream = f"stream-{socket.stream_number}"
        await output.start()
        await output.append("你好。")
        await socket.audio("你好。")
        sink._display_offset = 3
        await socket.face(0, 3, "你好。", stream=old_stream, angle=-12)
        await drain_native()
        assert output.pop_facial_cue() is None
        await socket.face(0, 3, "你好。", angle=12)
        await drain_native()
        assert output.pop_facial_cue()["targets"] == {"ParamAngleX": 12}
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_late_faces_cross_rotation_and_stay_private_until_draft_promotion():
    from fish_memory.companion.draft_pcm import PromotablePcmSink

    socket, sink = LateFacialSocket(), DeviceClockSink()
    arbiter = PlaybackArbiter()
    private = PromotablePcmSink(sink, 2000, playback_arbiter=arbiter)
    output = ApiTtsOutput(
        "http://localhost",
        sink=private,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    try:
        await output.start()
        await output.append("你好。")
        await socket.audio("你好。")
        await socket.incoming.put(json.dumps({"type": "segment_finished"}))
        await output.append("你好。")
        await socket.audio("你好。", 0)
        await drain_native()
        assert output._active_audio_range == (3, 6)
        await socket.face(3, 6, "你好。", angle=20)
        await socket.face(0, 3, "你好。", angle=10)
        await drain_native()
        assert output.pop_facial_cue() is None
        assert sink.calls == [] and output._active_audio_range == (3, 6)
        owner = await arbiter.acquire("answer", None)
        assert await private.promote(
            expected_end_offset=6,
            owner=owner,
            before_playback=lambda: arbiter.bind_tts(owner, output),
        )
        sink._display_offset = 3
        assert arbiter.current_tts.pop_facial_cue()["targets"] == {"ParamAngleX": 10}
        sink._played_offset = 3
        sink._display_offset = 6
        assert arbiter.current_tts.pop_facial_cue()["targets"] == {"ParamAngleX": 20}
        await output.interrupt()
        assert output.pop_facial_cue() is None
    finally:
        await output.close_transport()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_stream", [True, False])
async def test_implicit_append_ready_recovery_uses_correct_nonzero_text_base(
    same_stream,
):
    class RecoveringSocket(LateFacialSocket):
        append_count = 0

        async def send(self, value):
            message = json.loads(value)
            if message["type"] == "append":
                self.append_count += 1
                if self.append_count == 2:
                    # The append was sent before native could observe finish;
                    # server accepts it as a new provider task without start.
                    await self.incoming.put(json.dumps({"type": "finished"}))
                    if not same_stream:
                        self.stream_number += 1
                    await self.incoming.put(
                        json.dumps(
                            {
                                "type": "ready",
                                "facialStreamId": f"stream-{self.stream_number}",
                            }
                        )
                    )
            await super().send(value)

    socket, sink = RecoveringSocket(), DeviceClockSink()
    output = ApiTtsOutput(
        "http://localhost",
        sink=sink,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    try:
        await output.start(text_offset=20)
        await output.append("你好。")
        await socket.audio("你好。", 0)
        await drain_native()
        assert output._active_audio_range == (20, 23)
        await output.append("你好。")
        await drain_native()
        assert output._provider_task_active
        await socket.audio("你好。", 0)
        await socket.face(3 if same_stream else 0, 6 if same_stream else 3, "你好。")
        await drain_native()
        assert output._active_audio_range == (23, 26)
        assert output._audio_search_offset == 26
        sink._played_offset = 23
        sink._display_offset = 26
        assert output.pop_facial_cue()["targets"] == {"ParamAngleX": 12}
        assert sum(x["type"] == "start" for x in socket.sent) == 1
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_implicit_append_ready_clears_old_finished_before_waiting_for_new_audio():
    class RecoveringSocket(LateFacialSocket):
        append_count = 0

        async def send(self, value):
            message = json.loads(value)
            if message["type"] == "finish":
                self.sent.append(message)
                return  # The recovered provider has not produced its audio yet.
            if message["type"] == "append":
                self.append_count += 1
                if self.append_count == 2:
                    await self.incoming.put(json.dumps({"type": "finished"}))
                    await self.incoming.put(
                        json.dumps(
                            {
                                "type": "ready",
                                "facialStreamId": f"stream-{self.stream_number}",
                            }
                        )
                    )
            await super().send(value)

    socket, sink = RecoveringSocket(), DeviceClockSink()
    output = ApiTtsOutput(
        "http://localhost",
        sink=sink,
        facial=True,
        websocket_connector=lambda *args, **kwargs: _return_socket(socket),
    )
    finish = None
    try:
        await output.start()
        await output.append("第一句。")
        await socket.audio("第一句。")
        await drain_native()
        await output.append("第二句。")
        await drain_native()
        assert output._provider_task_active
        assert not output._finished.is_set()
        finish = asyncio.create_task(output.finish())
        await drain_native()
        assert not finish.done()
        # Even after PCM is queued, provider completion still belongs to this
        # recovered task; the earlier finished event cannot retire the answer.
        await socket.audio("第二句。")
        await drain_native()
        assert not finish.done()
        await socket.incoming.put(json.dumps({"type": "finished"}))
        await asyncio.wait_for(finish, 0.5)
        assert output._received_audio_end == len("第一句。第二句。")
    finally:
        if finish is not None and not finish.done():
            finish.cancel()
            await asyncio.gather(finish, return_exceptions=True)
        await output.close_transport()


class RefusingFinishSocket(FakeSocket):
    async def send(self, value: str) -> None:
        message = json.loads(value)
        if message["type"] == "finish":
            self.sent.append(message)
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Please ensure input text is valid.",
                        "failure": "voice:rejected",
                    }
                )
            )
            return
        await super().send(value)


@pytest.mark.asyncio
async def test_speech_error_keeps_the_reason_the_api_gave() -> None:
    from fish_memory import failures

    socket = RefusingFinishSocket()

    async def connector(_: str) -> FakeSocket:
        return socket

    output = ApiTtsOutput(
        "http://127.0.0.1:8000", sink=FakeSink(), websocket_connector=connector
    )
    await output.start()
    await output.append("helello.")
    with pytest.raises(Exception) as raised:
        await output.finish()
    assert failures.failure_of(raised.value) == failures.ServiceFailure(
        "voice", "rejected"
    )
    await output.close()
