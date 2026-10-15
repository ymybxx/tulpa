"""Facial tests use the active character and the runtime's existing TTS service."""

import asyncio
import json
from collections import deque

import httpx
import pytest
from websockets.exceptions import ConnectionClosedError

from fish_memory.facial_test import runtime_speech

API_URL = "http://127.0.0.1:54321"
API_KEY = "private-runtime-token"
FISH_IDS = [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
]


def mock_http(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        runtime_speech.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


async def test_load_uses_current_character_and_reloads_after_switch(monkeypatch):
    current = 0
    requests = []
    voices = [
        {
            "provider": "dashscope",
            "config": {
                "model": "qwen-audio-3.0-tts-flash",
                "voice": "longanxiaoxin",
                "kind": "preset",
            },
        },
        {
            "provider": "local",
            "config": {
                "model": "local-fixture",
                "voice": "default",
                "kind": "local",
            },
        },
    ]

    def respond(request):
        requests.append(request)
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        if request.url.path == "/v1/fishes/assistant":
            return httpx.Response(200, json={"fishId": FISH_IDS[current]})
        assert request.url.path == f"/v1/fishes/{FISH_IDS[current]}/voice"
        return httpx.Response(200, json=voices[current])

    mock_http(monkeypatch, respond)
    cloud = await runtime_speech.load_runtime_speech(API_URL + "/", API_KEY)
    assert (cloud.provider, cloud.model, cloud.voice, cloud.sample_rate) == (
        "dashscope",
        "qwen-audio-3.0-tts-flash",
        "longanxiaoxin",
        24_000,
    )
    assert API_KEY not in repr(cloud)
    current = 1
    local = await runtime_speech.load_runtime_speech(API_URL, API_KEY)
    assert (local.provider, local.model, local.voice, local.sample_rate) == (
        "local",
        "local-fixture",
        "default",
        16_000,
    )
    assert len(requests) == 4


@pytest.mark.parametrize(
    "binding,voice",
    [
        ({"fishId": None}, {}),
        ({"fishId": "invalid"}, {}),
        ([], {}),
        ({"fishId": FISH_IDS[0]}, {"provider": "auto", "config": {}}),
        ({"fishId": FISH_IDS[0]}, {"provider": [], "config": {}}),
        (
            {"fishId": FISH_IDS[0]},
            {
                "provider": "local",
                "config": {
                    "model": "local",
                    "voice": "default",
                    "kind": "preset",
                },
            },
        ),
        (
            {"fishId": FISH_IDS[0]},
            {
                "provider": "dashscope",
                "config": {
                    "model": " ",
                    "voice": "default",
                    "kind": "preset",
                },
            },
        ),
        (
            {"fishId": FISH_IDS[0]},
            {
                "provider": "dashscope",
                "config": {
                    "model": "model",
                    "voice": 12,
                    "kind": "custom",
                },
            },
        ),
        (
            {"fishId": FISH_IDS[0]},
            {
                "provider": "dashscope",
                "config": {
                    "model": "model",
                    "voice": "voice",
                    "kind": [],
                },
            },
        ),
    ],
)
async def test_invalid_runtime_responses_are_safe(monkeypatch, binding, voice):
    mock_http(
        monkeypatch,
        lambda req: httpx.Response(
            200,
            json=binding if req.url.path.endswith("assistant") else voice,
        ),
    )
    with pytest.raises(ValueError) as error:
        await runtime_speech.load_runtime_speech(API_URL, API_KEY)
    assert API_KEY not in str(error.value)


@pytest.mark.parametrize("failure", ["http", "connection", "json"])
async def test_http_failures_hide_credentials(monkeypatch, failure):
    def respond(request):
        if failure == "connection":
            raise httpx.ConnectError(API_KEY, request=request)
        return httpx.Response(401 if failure == "http" else 200, text=API_KEY)

    mock_http(monkeypatch, respond)
    with pytest.raises(ValueError) as error:
        await runtime_speech.load_runtime_speech(API_URL, API_KEY)
    assert API_KEY not in str(error.value)
    assert error.value.__suppress_context__


class FakeSocket:
    def __init__(self, events):
        self.events = deque(events)
        self.sent = []
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def recv(self):
        if not self.events:
            self.waiting.set()
            await asyncio.Future()
        event = self.events.popleft()
        if isinstance(event, Exception):
            raise event
        return json.dumps(event) if isinstance(event, (dict, list)) else event


def speech():
    return runtime_speech.RuntimeSpeech(
        API_URL,
        API_KEY,
        "dashscope",
        "chosen-model",
        "chosen-voice",
        24_000,
    )


def mock_socket(monkeypatch, events):
    socket = FakeSocket(events)

    def connect(url, **kwargs):
        assert url == "ws://127.0.0.1:54321/v1/tts/stream"
        assert kwargs["additional_headers"] == {
            "Authorization": f"Bearer {API_KEY}",
        }
        return socket

    monkeypatch.setattr(runtime_speech.websockets, "connect", connect)
    return socket


async def test_stream_protocol_preserves_voice_and_pcm_and_closes(monkeypatch):
    socket = mock_socket(
        monkeypatch,
        [
            {"type": "connecting"},
            {"type": "ready", "sampleRate": 24_000},
            {"type": "sentence_begin"},
            {"type": "audio"},
            b"\x00\x01",
            {"type": "audio"},
            b"\x02\x03",
            {"type": "finished"},
        ],
    )
    assert [pcm async for pcm in runtime_speech.runtime_audio(speech(), "辛苦啦")] == [
        b"\x00\x01",
        b"\x02\x03",
    ]
    assert socket.sent == [
        {
            "type": "start",
            "model": "chosen-model",
            "voice": "chosen-voice",
            "audio_format": "pcm",
            "sample_rate": 24_000,
        },
        {"type": "append", "text": "辛苦啦"},
        {"type": "finish"},
    ]
    assert socket.closed


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "error", "message": API_KEY}],
        [{"type": "ready", "sampleRate": 16_000}],
        [{"type": "ready"}],
        [{"type": "finished"}],
        [{"type": "ready", "sampleRate": 24_000}, {"type": "finished"}],
        [
            {"type": "ready", "sampleRate": 24_000},
            b"\x00\x01",
            {"type": "error", "message": API_KEY},
        ],
        [
            {"type": "ready", "sampleRate": 24_000},
            b"\x00\x01",
            {"type": "finished", "stopped": True},
        ],
        [b"\x00\x01"],
        [API_KEY],
        [[]],
        [ConnectionClosedError(None, None)],
    ],
)
async def test_stream_failure_closes_and_does_not_expose_token(monkeypatch, events):
    socket = mock_socket(monkeypatch, events)
    with pytest.raises(ValueError) as error:
        _ = [pcm async for pcm in runtime_speech.runtime_audio(speech(), "你好")]
    assert API_KEY not in str(error.value)
    assert socket.closed


@pytest.mark.parametrize(
    "timeout_attribute",
    [
        "_RECEIVE_TIMEOUT_SECONDS",
        "_STREAM_TIMEOUT_SECONDS",
    ],
)
async def test_stream_timeout_closes(monkeypatch, timeout_attribute):
    socket = mock_socket(monkeypatch, [])
    monkeypatch.setattr(runtime_speech, timeout_attribute, 0.01)
    with pytest.raises(ValueError, match="超时"):
        _ = [pcm async for pcm in runtime_speech.runtime_audio(speech(), "你好")]
    assert socket.closed


async def test_closing_partial_audio_stream_closes_socket(monkeypatch):
    socket = mock_socket(
        monkeypatch,
        [
            {"type": "ready", "sampleRate": 24_000},
            b"\x00\x01",
        ],
    )
    stream = runtime_speech.runtime_audio(speech(), "你好")
    assert await anext(stream) == b"\x00\x01"
    await stream.aclose()
    assert socket.closed


async def test_cancellation_closes_stream(monkeypatch):
    socket = mock_socket(monkeypatch, [])

    async def consume():
        return [pcm async for pcm in runtime_speech.runtime_audio(speech(), "你好")]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(socket.waiting.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed
