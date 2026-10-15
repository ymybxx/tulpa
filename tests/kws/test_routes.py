from fastapi import FastAPI
from fastapi.testclient import TestClient

from fish_memory.kws.models import Detection
from fish_memory.kws.routes import kws_router


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.reset_count = 0

    def accept_pcm(self, pcm_bytes: bytes) -> list[Detection]:
        if pcm_bytes:
            return [Detection("小鱼", 10, 0.5)]
        return []

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class FakeKwsService:
    available = True
    default_wake_word = "小鱼"
    last_connection: FakeConnection | None = None

    def status(self) -> dict[str, object]:
        return {
            "available": True,
            "wakeWord": self.default_wake_word,
            "sampleRate": 16_000,
            "channels": 1,
            "chunkSize": 8,
            "error": None,
        }

    async def open_stream(self, wake_word: str | None) -> FakeConnection:
        assert wake_word == "小鱼"
        self.last_connection = FakeConnection()
        return self.last_connection


def build_app() -> tuple[FastAPI, FakeKwsService]:
    app = FastAPI()
    service = FakeKwsService()
    app.state.kws_service = service
    app.include_router(kws_router)
    return app, service


def test_kws_websocket_requires_start_message() -> None:
    app, _service = build_app()

    with TestClient(app).websocket_connect("/v1/kws/stream") as socket:
        socket.send_bytes(b"\x00\x00" * 160)
        assert socket.receive_json()["type"] == "error"


def test_kws_websocket_returns_ready_stats_and_detection() -> None:
    app, service = build_app()

    with TestClient(app).websocket_connect("/v1/kws/stream") as socket:
        socket.send_json(
            {
                "type": "start",
                "wakeWord": "小鱼",
                "sampleRate": 16_000,
                "channels": 1,
                "format": "s16le",
            }
        )
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x00\x00" * 160)
        messages = [socket.receive_json(), socket.receive_json()]

    assert {message["type"] for message in messages} == {"stats", "detected"}
    assert service.last_connection is not None
    assert service.last_connection.closed is True


def test_kws_websocket_rejects_wrong_audio_format() -> None:
    app, _service = build_app()

    with TestClient(app).websocket_connect("/v1/kws/stream") as socket:
        socket.send_json(
            {
                "type": "start",
                "wakeWord": "小鱼",
                "sampleRate": 8_000,
                "channels": 1,
                "format": "s16le",
            }
        )
        assert socket.receive_json()["type"] == "error"


def test_kws_status_exposes_runtime_capability() -> None:
    app, _service = build_app()

    response = TestClient(app).get("/v1/kws/status")

    assert response.status_code == 200
    assert response.json()["sampleRate"] == 16_000
