from __future__ import annotations

import asyncio
import json
import stat
import struct
import sys
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import pytest

import fish_memory.companion.shared_asr as shared_module
from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.shared_asr import (
    SharedAsrClient,
    SharedAsrResponse,
    SharedAsrServer,
)


class Recognizer:
    model = "sensevoice"

    def __init__(self, responses: list[AsrUpdate | None]) -> None:
        self.responses = responses
        self.calls: list[bytes] = []
        self.buffered = 0

    def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate | None:
        self.calls.append(pcm)
        return self.responses.pop(0) if self.responses else None

    def buffered_pcm_bytes(self) -> int:
        return self.buffered


def _socket_path() -> Path:
    return Path(tempfile.gettempdir()) / f"fma-{uuid4().hex}.sock"


@pytest.mark.asyncio
async def test_shared_asr_recognizes_on_the_bound_authoritative_model():
    path = _socket_path()
    recognizer = Recognizer([AsrUpdate("本地系统声音", True, "Speech")])
    server = SharedAsrServer(path)
    server.bind(recognizer)
    try:
        result = await server._recognize(b"\x01\x00" * 160)
        assert result.status == "ok"
        assert result.text == "本地系统声音"
        assert result.event == "Speech"
        assert recognizer.calls == [b"\x01\x00" * 160]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_waits_for_low_priority_slot_without_parallel_decode():
    path = _socket_path()
    recognizer = Recognizer([None, None, AsrUpdate("ready", True, "Speech")])
    server = SharedAsrServer(path)
    server.bind(recognizer)
    try:
        result = await server._recognize(b"\x01\x00")
        assert result.status == "ok"
        assert result.text == "ready"
        assert len(recognizer.calls) == 3
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_does_not_decode_while_user_pcm_is_buffered():
    path = _socket_path()
    recognizer = Recognizer([AsrUpdate("after user", True, "Speech")])
    recognizer.buffered = 320
    server = SharedAsrServer(path)
    server.bind(recognizer)
    try:
        pending = asyncio.create_task(server._recognize(b"\x01\x00"))
        await asyncio.sleep(0.06)
        assert recognizer.calls == []
        recognizer.buffered = 0
        result = await pending
        assert result.status == "ok"
        assert recognizer.calls == [b"\x01\x00"]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_reports_busy_without_running_a_second_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(shared_module, "_BUSY_WAIT_SECONDS", 0.001)
    monkeypatch.setattr(shared_module, "_BUSY_RETRY_SECONDS", 0.001)
    path = _socket_path()
    recognizer = Recognizer([])
    server = SharedAsrServer(path)
    server.bind(recognizer)
    try:
        result = await server._recognize(b"\x01\x00")
        assert result.status == "busy"
        assert len(recognizer.calls) >= 1
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_unavailable_does_not_claim_pcm():
    path = _socket_path()
    server = SharedAsrServer(path)
    try:
        result = await server._recognize(b"\x01\x00")
        assert result.status == "unavailable"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_rejects_oversized_pcm_before_connecting():
    with pytest.raises(ValueError, match="length"):
        await SharedAsrClient(_socket_path()).recognize(
            b"\x01\x00" * (shared_module.MAX_PCM_BYTES // 2 + 1)
        )


@pytest.mark.asyncio
async def test_shared_asr_start_refuses_to_unlink_a_regular_file():
    path = _socket_path()
    path.write_text("keep")
    server = SharedAsrServer(path, transport="unix")
    try:
        with pytest.raises(RuntimeError, match="non-socket"):
            await server.start()
        await server.close()
        assert path.read_text() == "keep"
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket mode bits")
async def test_shared_asr_server_uses_a_private_owned_socket_path(monkeypatch):
    path = _socket_path()

    class FakeServer:
        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def start_unix_server(_handler, *, path: str):
        await asyncio.to_thread(Path(path).touch)
        return FakeServer()

    monkeypatch.setattr(asyncio, "start_unix_server", start_unix_server)
    server = SharedAsrServer(path, transport="unix")
    await server.start()
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        await server.close()
    assert not path.exists()


@pytest.mark.asyncio
async def test_shared_asr_closed_peer_never_starts_decode():
    recognizer = Recognizer([AsrUpdate("unused", True, "Speech")])
    server = SharedAsrServer(_socket_path())
    server.bind(recognizer)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    try:
        assert await server._recognize_connected(reader, b"\x01\x00") is None
        assert recognizer.calls == []
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_disconnect_cancels_busy_retries(monkeypatch):
    # A long retry interval makes the completion deadline prove cancellation,
    # rather than letting a retry observe some unrelated recognizer state.
    monkeypatch.setattr(shared_module, "_BUSY_RETRY_SECONDS", 1.0)
    attempted = asyncio.Event()
    loop = asyncio.get_running_loop()

    class BusyRecognizer(Recognizer):
        def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate | None:
            self.calls.append(pcm)
            loop.call_soon_threadsafe(attempted.set)
            return None

    recognizer = BusyRecognizer([])
    server = SharedAsrServer(_socket_path())
    server.bind(recognizer)
    reader = asyncio.StreamReader()
    pending = asyncio.create_task(server._recognize_connected(reader, b"\x01\x00"))
    try:
        await asyncio.wait_for(attempted.wait(), timeout=1.0)
        reader.feed_eof()
        assert await asyncio.wait_for(pending, timeout=0.2) is None
        assert recognizer.calls == [b"\x01\x00"]
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_disconnect_removes_request_queued_behind_native_decode():
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class BlockingRecognizer(Recognizer):
        def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate:
            self.calls.append(pcm)
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=2.0)
            return AsrUpdate("finished", True, "Speech")

    recognizer = BlockingRecognizer([])
    server = SharedAsrServer(_socket_path())
    server.bind(recognizer)
    first_reader, retired_reader = asyncio.StreamReader(), asyncio.StreamReader()
    first = asyncio.create_task(server._recognize_connected(first_reader, b"\x01\x00"))
    retired = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        retired = asyncio.create_task(
            server._recognize_connected(retired_reader, b"\x02\x00")
        )
        # The first native inference holds the request lock; the second request
        # must disappear on disconnect instead of decoding after it is released.
        await asyncio.sleep(0)
        retired_reader.feed_eof()
        assert await asyncio.wait_for(retired, timeout=0.2) is None
        release.set()
        response = await asyncio.wait_for(first, timeout=1.0)
        assert response is not None and response.status == "ok"
        assert recognizer.calls == [b"\x01\x00"]
    finally:
        release.set()
        for task in (first, retired):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, retired) if task is not None),
            return_exceptions=True,
        )
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_disconnect_discards_in_flight_native_result():
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class BlockingRecognizer(Recognizer):
        def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate:
            self.calls.append(pcm)
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=2.0)
            return AsrUpdate("stale", True, "Speech")

    recognizer = BlockingRecognizer([])
    server = SharedAsrServer(_socket_path())
    server.bind(recognizer)
    reader = asyncio.StreamReader()
    pending = asyncio.create_task(server._recognize_connected(reader, b"\x01\x00"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        reader.feed_eof()
        assert await asyncio.wait_for(pending, timeout=0.2) is None
        assert recognizer.calls == [b"\x01\x00"]
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await server.close()


@pytest.mark.asyncio
async def test_shared_asr_preserves_native_alignment_without_another_decode():
    native = AsrUpdate(
        "今天吃什么？",
        True,
        "Speech",
        ("今天", "吃", "什么", "？"),
        (0.12, 0.36, 0.48, 0.72),
    )
    recognizer = Recognizer([native])
    server = SharedAsrServer(_socket_path())
    server.bind(recognizer)
    pcm = b"\0\0" * 16_000
    try:
        response = await server._recognize(pcm)
        received = shared_module._parse_response(
            shared_module._encode_response(response)
        )
        assert received == SharedAsrResponse(
            "ok", native.text, native.event, native.tokens, native.timestamps
        )
        assert recognizer.calls == [pcm]
    finally:
        await server.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"tokens": ["好"]},
        {"timestamps": [0.1]},
        {"tokens": ["好"], "timestamps": [0.1, 0.2]},
        {"tokens": ["好"], "timestamps": [float("nan")]},
        {"tokens": ["好"], "timestamps": [float("inf")]},
        {"tokens": ["好"], "timestamps": [-0.1]},
        {"tokens": ["好"], "timestamps": [True]},
        {"tokens": ["好"], "timestamps": ["0.1"]},
        {"tokens": ["好", "吧"], "timestamps": [0.3, 0.2]},
        {"tokens": ["好"], "timestamps": None},
        {"tokens": {"text": "好"}, "timestamps": [0.1]},
        {"tokens": ["好", 1], "timestamps": [0.1, 0.2]},
        {"tokens": ["好" * 257], "timestamps": [0.1]},
    ],
)
def test_shared_asr_old_or_bad_alignment_keeps_transcript(metadata):
    payload = json.dumps(
        {
            "status": "ok",
            "text": "好，今天想吃什么？",
            "event": "Speech",
            **metadata,
        }
    ).encode()
    assert shared_module._parse_response(payload) == SharedAsrResponse(
        "ok", "好，今天想吃什么？", "Speech"
    )


@pytest.mark.parametrize(
    "text",
    ["好" * 4001, "😀" * 4000, "\x01" * 4000],
    ids=["long-cjk", "multibyte-emoji", "control-characters"],
)
def test_shared_asr_response_bounds_drop_alignment_when_text_is_shortened(text):
    payload = shared_module._encode_response(
        SharedAsrResponse(
            "ok",
            text,
            "😀" * 100,
            ("好",),
            (0.2,),
        )
    )
    assert len(payload) <= shared_module._MAX_RESPONSE_BYTES
    received = shared_module._parse_response(payload)
    assert text.startswith(received.text) and received.text
    assert len(received.text) < len(text)
    assert received.tokens == received.timestamps == ()
    assert received.event == "😀" * 100


def test_shared_asr_oversized_alignment_is_dropped_before_text():
    transcript = "原文完整保留。"
    payload = shared_module._encode_response(
        SharedAsrResponse(
            "ok",
            transcript,
            "Speech",
            ("好",) * 4096,
            (0.2,) * 4096,
        )
    )
    assert len(payload) <= shared_module._MAX_RESPONSE_BYTES
    assert shared_module._parse_response(payload) == SharedAsrResponse(
        "ok", transcript, "Speech"
    )


@pytest.mark.asyncio
async def test_shared_asr_truncated_native_result_cannot_keep_full_token_times():
    native = AsrUpdate("好" * 4001, True, "Speech", ("好",), (0.12,))
    server = SharedAsrServer(_socket_path())
    server.bind(Recognizer([native]))
    try:
        response = await server._recognize(b"\0\0")
        assert response.text == "好" * 4000
        assert response.tokens == response.timestamps == ()
    finally:
        await server.close()


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.asyncio
async def test_shared_asr_client_reads_old_and_new_response_frames(
    monkeypatch, aligned
):
    response = SharedAsrResponse(
        "ok",
        "好",
        "Speech",
        ("好",) if aligned else (),
        (0.12,) if aligned else (),
    )
    payload = shared_module._encode_response(response)
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", len(payload)) + payload)
    reader.feed_eof()

    class Writer:
        def __init__(self):
            self.written = bytearray()
            self.closed = False

        def write(self, data):
            self.written.extend(data)

        async def drain(self):
            pass

        def close(self):
            self.closed = True

        async def wait_closed(self):
            pass

    writer = Writer()

    async def connect(_path):
        return reader, writer

    # This isolates response framing from the transport; supply the Unix stub
    # even on Windows, whose asyncio module does not expose that API.
    monkeypatch.setattr(asyncio, "open_unix_connection", connect, raising=False)
    pcm = b"\0\0" * 160
    assert (
        await SharedAsrClient(_socket_path(), transport="unix").recognize(pcm)
        == response
    )
    assert (
        writer.written
        == shared_module._HEADER.pack(shared_module._MAGIC, len(pcm)) + pcm
    )
    assert writer.closed


@pytest.mark.parametrize("status", ["busy", "unavailable", "invalid", "error"])
def test_shared_asr_non_success_never_carries_alignment(status):
    payload = json.dumps(
        {
            "status": status,
            "text": "",
            "event": "",
            "tokens": ["好"],
            "timestamps": [0.12],
        }
    ).encode()
    assert shared_module._parse_response(payload) == SharedAsrResponse(status)


def test_shared_asr_unknown_protocol_fields_are_still_rejected():
    with pytest.raises(ValueError, match="invalid fields"):
        shared_module._parse_response(
            json.dumps(
                {
                    "status": "ok",
                    "text": "好",
                    "event": "Speech",
                    "unexpected": True,
                }
            ).encode()
        )
