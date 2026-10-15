from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import fish_memory.companion.environment_asr as environment
from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.shared_asr import SharedAsrClient, SharedAsrServer


@pytest.mark.asyncio
async def test_media_asr_handoff_uses_same_socket_and_loads_only_on_audio(monkeypatch):
    root = Path(tempfile.gettempdir()) if sys.platform == "win32" else Path("/tmp")
    path = root / f"environment-asr-{uuid4().hex}.sock"
    monkeypatch.setenv("FISH_MEMORY_SHARED_ASR_SOCKET", str(path))
    models = []
    calls = []

    class Decoder:
        model = "sensevoice"

        def __init__(self, config):
            assert config.num_threads == 1
            models.append(self)

        def buffered_pcm_bytes(self):
            return 0

        def try_recognize_snapshot_update(self, pcm):
            calls.append(pcm)
            return AsrUpdate(
                "原来的对白", True, "Speech", ("原来", "的对白"), (0.1, 0.5)
            )

    monkeypatch.setattr(environment, "StreamingAsr", Decoder)
    monkeypatch.setattr(
        environment,
        "AsrModelDownloader",
        lambda _: SimpleNamespace(ensure=lambda: None),
    )
    stop = asyncio.Event()
    client = SharedAsrClient(path)
    voice = SharedAsrServer(path)
    voice.bind(
        SimpleNamespace(
            model="sensevoice",
            buffered_pcm_bytes=lambda: 0,
            try_recognize_snapshot_update=lambda _: AsrUpdate(
                "语音模式", True, "Speech"
            ),
        )
    )
    await voice.start()
    assert (await client.recognize(b"\x05\x00" * 100)).text == "语音模式"
    await voice.close()
    ready = asyncio.Event()
    start_server = SharedAsrServer.start

    async def start(self):
        await start_server(self)
        ready.set()

    monkeypatch.setattr(SharedAsrServer, "start", start)
    worker = asyncio.create_task(
        environment.run_environment_asr(SimpleNamespace(), stop_event=stop)
    )
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert models == []
        for _ in range(2):
            response = await client.recognize(b"\x05\x00" * 100)
            assert response.text == "原来的对白"
            assert response.timestamps == (0.1, 0.5)
        assert len(models) == 1
        assert len(calls) == 2
    finally:
        stop.set()
        await worker
    assert not path.exists()
    resumed = SharedAsrServer(path)
    resumed.bind(
        SimpleNamespace(
            model="sensevoice",
            buffered_pcm_bytes=lambda: 0,
            try_recognize_snapshot_update=lambda _: AsrUpdate(
                "恢复交流", True, "Speech"
            ),
        )
    )
    try:
        await resumed.start()
        assert (await client.recognize(b"\x05\x00" * 100)).text == "恢复交流"
        assert len(calls) == 2  # Retired background recognizer receives no more work.
    finally:
        await resumed.close()
