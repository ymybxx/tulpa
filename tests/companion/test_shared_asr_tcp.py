from __future__ import annotations

import asyncio
import json
import secrets
import threading
from types import SimpleNamespace

import pytest

from fish_memory.companion import shared_asr as shared
from fish_memory.companion.asr import AsrUpdate


def recognizer(calls):
    def recognize(pcm):
        calls.append(pcm)
        return AsrUpdate("本地测试", True, "Speech", ("本地", "测试"), (0.1, 0.3))

    return SimpleNamespace(
        model="test",
        buffered_pcm_bytes=lambda: 0,
        try_recognize_snapshot_update=recognize,
    )


@pytest.mark.asyncio
async def test_tcp_roundtrip_authenticates_and_reuses_one_model(tmp_path):
    path = tmp_path / "中文 endpoint.json"
    calls = []
    server = shared.SharedAsrServer(path, transport="tcp")
    server.bind(recognizer(calls))
    try:
        await server.start()
        metadata = json.loads(path.read_text())
        assert 0 < metadata["port"] < 65536
        assert len(metadata["token"]) == 64
        assert server._server.sockets[0].getsockname()[0] == "127.0.0.1"
        client = shared.SharedAsrClient(path, transport="tcp")
        for _ in range(2):
            result = await client.recognize(b"\0\0" * 160)
            assert result.text == "本地测试"
            assert result.timestamps == (0.1, 0.3)
        assert len(calls) == 2
    finally:
        await server.close()
    assert not path.exists()


@pytest.mark.asyncio
async def test_wrong_token_never_reaches_recognizer(tmp_path):
    path = tmp_path / "endpoint.json"
    wrong = tmp_path / "wrong.json"
    calls = []
    server = shared.SharedAsrServer(path, transport="tcp")
    server.bind(recognizer(calls))
    try:
        await server.start()
        endpoint = json.loads(path.read_text())
        endpoint["token"] = "00" * 32
        wrong.write_text(json.dumps(endpoint))
        with pytest.raises(ConnectionError, match="authentication"):
            await shared.SharedAsrClient(wrong, transport="tcp").recognize(b"\0\0")
        assert calls == []
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_client_never_sends_pcm_to_process_reusing_a_stale_port(tmp_path):
    received = []
    finished = asyncio.Event()

    async def impostor(reader, writer):
        try:
            received.append(await reader.readexactly(36))
            writer.write(bytes(64))
            await writer.drain()
            received.append(await reader.read())
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(impostor, "127.0.0.1", 0)
    path = tmp_path / "endpoint.json"
    shared._write_endpoint(
        path, server.sockets[0].getsockname()[1], secrets.token_bytes(32)
    )
    try:
        with pytest.raises(ConnectionError, match="authentication"):
            await shared.SharedAsrClient(path, transport="tcp").recognize(b"\x12\x34")
        await asyncio.wait_for(finished.wait(), 2)
        assert len(received[0]) == 36
        assert received[1] == b""
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tcp_refuses_to_replace_live_owner_or_unrelated_file(tmp_path):
    path = tmp_path / "endpoint.json"
    owner = shared.SharedAsrServer(path, transport="tcp")
    contender = shared.SharedAsrServer(path, transport="tcp")
    try:
        await owner.start()
        metadata = path.read_bytes()
        with pytest.raises(RuntimeError, match="already in use"):
            await contender.start()
        assert path.read_bytes() == metadata
        await contender.close()
        assert path.read_bytes() == metadata
    finally:
        await owner.close()
        await contender.close()
    path.write_text("keep me")
    other = shared.SharedAsrServer(path, transport="tcp")
    try:
        with pytest.raises(ValueError, match="invalid"):
            await other.start()
        assert path.read_text() == "keep me"
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_tcp_cleanup_does_not_delete_new_owner_endpoint(tmp_path):
    path = tmp_path / "endpoint.json"
    server = shared.SharedAsrServer(path, transport="tcp")
    await server.start()
    replacement = secrets.token_bytes(32)
    shared._write_endpoint(path, 12345, replacement)
    await server.close()
    assert shared._read_endpoint(path) == (12345, replacement)


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_endpoint_publication_and_cleans_up(
    tmp_path, monkeypatch
):
    path = tmp_path / "endpoint.json"
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    publish = shared._write_endpoint

    def paused_publish(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(2)
        publish(*args)

    monkeypatch.setattr(shared, "_write_endpoint", paused_publish)
    server = shared.SharedAsrServer(path, transport="tcp")
    start = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        start.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert not path.exists()
    finally:
        release.set()
        await server.close()


@pytest.mark.parametrize(
    "value",
    [
        {"version": 1, "port": True, "token": "00" * 32},
        {"version": 1, "port": 1234, "token": "xx" * 32},
        {"version": 1, "port": 1234, "token": "00" * 32, "host": "remote.example"},
        {"version": True, "port": 1234, "token": "00" * 32},
    ],
)
def test_tcp_endpoint_is_bounded_and_cannot_select_a_remote_host(tmp_path, value):
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid"):
        shared._read_endpoint(path)
    path.write_text(" " * 1024)
    with pytest.raises(ValueError, match="too large"):
        shared._read_endpoint(path)
