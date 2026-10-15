import asyncio
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from fish_memory.tts.catalog import get_tts_catalog
from fish_memory.tts.models import TtsAppendRequest, TtsRequest, TtsStartRequest
from fish_memory.tts.provider import (
    build_cancel_task,
    build_continue_task,
    build_finish_task,
    build_run_task,
    parse_server_event,
)


def test_tts_catalog_exposes_preset_voices() -> None:
    catalog = get_tts_catalog()

    assert [model.id for model in catalog] == ["qwen-audio-3.0-tts-flash"]
    flash = catalog[0]
    assert any(voice.id == "longanhuan_v3.6" for voice in flash.voices)


def test_starlette_post_disconnect_runtime_error_is_not_a_provider_failure() -> None:
    from fish_memory.tts.routes import _is_disconnected_websocket_error

    assert _is_disconnected_websocket_error(
        RuntimeError("WebSocket is not connected. Need to call 'accept' first.")
    )
    assert not _is_disconnected_websocket_error(RuntimeError("provider failed"))


@pytest.mark.asyncio
async def test_tts_route_treats_post_disconnect_runtime_error_as_normal_close() -> None:
    from fish_memory.tts.routes import tts_stream

    class State:
        settings = object()

    class App:
        state = State()

    class DisconnectedSocket:
        app = App()

        def __init__(self) -> None:
            self.accepted = False
            self.closed = False
            self.events: list[dict[str, object]] = []

        async def accept(self) -> None:
            self.accepted = True

        async def receive_json(self) -> dict[str, object]:
            raise RuntimeError(
                "WebSocket is not connected. Need to call 'accept' first."
            )

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

        async def close(self, code: int = 1000) -> None:
            del code
            self.closed = True

    socket = DisconnectedSocket()

    await tts_stream(socket)  # type: ignore[arg-type]

    assert socket.accepted is True
    assert socket.closed is True
    assert socket.events == []


def test_streaming_messages_separate_task_parameters_from_text_appends() -> None:
    start = TtsStartRequest(
        model="cosyvoice-v3-flash",
        voice="longanyang",
        audio_format="pcm",
        sample_rate=22_050,
    )
    append = TtsAppendRequest(text="第一句。")

    assert start.model == "cosyvoice-v3-flash"
    assert start.voice == "longanyang"
    assert append.text == "第一句。"


def test_dashscope_task_messages_use_duplex_protocol() -> None:
    request = TtsRequest(
        model="cosyvoice-v3-flash",
        voice="longanyang",
        text="你好，世界。",
    )

    run_task = build_run_task(request, task_id="task-1")
    continue_task = build_continue_task("task-1", request.text)
    finish_task = build_finish_task("task-1")

    assert run_task["header"] == {
        "action": "run-task",
        "task_id": "task-1",
        "streaming": "duplex",
    }
    assert run_task["payload"]["task"] == "tts"
    assert run_task["payload"]["function"] == "SpeechSynthesizer"
    assert run_task["payload"]["model"] == request.model
    assert run_task["payload"]["parameters"]["voice"] == request.voice
    assert run_task["payload"]["parameters"]["format"] == "pcm"
    assert continue_task == {
        "header": {
            "action": "continue-task",
            "task_id": "task-1",
            "streaming": "duplex",
        },
        "payload": {"input": {"text": request.text}},
    }
    assert finish_task["header"]["action"] == "finish-task"
    assert build_cancel_task("task-1") == {
        "header": {
            "action": "finish-task",
            "task_id": "task-1",
            "streaming": "duplex",
        },
        "payload": {"input": {"directive": "cancel"}},
    }


def test_server_event_parser_extracts_audio_and_terminal_events() -> None:
    synthesis = parse_server_event(
        {
            "header": {"event": "result-generated"},
            "payload": {
                "output": {
                    "type": "sentence-synthesis",
                    "sentence": "你好。",
                    "sentence_id": 2,
                }
            },
        }
    )
    finished = parse_server_event(
        {
            "header": {"event": "task-finished"},
            "payload": {"output": {"finish_reason": "stop"}},
        }
    )

    assert synthesis.kind == "audio"
    assert synthesis.sentence_index == 2
    assert synthesis.text == "你好。"
    assert finished.kind == "finished"


def test_server_event_parser_extracts_official_sentence_shape() -> None:
    synthesis = parse_server_event(
        {
            "header": {"event": "result-generated"},
            "payload": {
                "output": {
                    "type": "sentence-synthesis",
                    "sentence": {"index": 3},
                    "original_text": "你好。",
                }
            },
        }
    )

    assert synthesis.kind == "audio"
    assert synthesis.sentence_index == 3
    assert synthesis.text == "你好。"


@pytest.mark.asyncio
async def test_tts_session_serializes_commands_before_sending() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send(self, message: str) -> None:
            self.messages.append(message)

    from fish_memory.tts.provider import DashScopeTtsSession

    session = DashScopeTtsSession(websocket_url="wss://example.invalid", api_key="key")
    socket = FakeSocket()
    session._socket = socket

    await session.send({"hello": "世界"})

    assert json.loads(socket.messages[0]) == {"hello": "世界"}


def test_tts_session_generates_a_new_task_id_for_each_reused_task() -> None:
    from fish_memory.tts.provider import DashScopeTtsSession

    session = DashScopeTtsSession(websocket_url="wss://example.invalid", api_key="key")

    first_task_id = session.begin_task()
    second_task_id = session.begin_task()

    assert first_task_id != second_task_id
    assert session.task_id == second_task_id


def test_tts_accepts_commands_queued_after_provider_terminal_event() -> None:
    from fish_memory.tts.routes import _parse_client_task_message

    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        audio_format="pcm",
        sample_rate=22_050,
    )

    next_request, initial_text, is_final, stale = _parse_client_task_message(
        {"type": "append", "text": "下一段"},
        request,
    )
    assert next_request is request
    assert initial_text == "下一段"
    assert is_final is False
    assert stale is False

    next_request, initial_text, is_final, stale = _parse_client_task_message(
        {"type": "finish"},
        request,
    )
    assert next_request is None
    assert initial_text is None
    assert is_final is True
    assert stale is True


@pytest.mark.asyncio
async def test_control_loop_forwards_append_and_finish_commands() -> None:
    from fish_memory.tts.routes import _wait_for_control

    class FakeBrowser:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    {"type": "append", "text": "第一句。"},
                    {"type": "finish"},
                ]
            )
            self.events: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            return next(self.messages)

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    class FakeSession:
        task_id = "task-1"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, message: dict[str, object]) -> None:
            self.messages.append(message)

    browser = FakeBrowser()
    session = FakeSession()

    await _wait_for_control(
        browser,
        session,
        asyncio.Event(),
        {"text_sent_at": None},
    )

    assert session.messages == [
        build_continue_task("task-1", "第一句。"),
        build_finish_task("task-1"),
    ]


@pytest.mark.asyncio
async def test_control_loop_rotates_idle_provider_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fish_memory.tts.routes as routes

    monkeypatch.setattr(routes, "TTS_IDLE_ROTATION_INTERVAL_SECONDS", 0.001)

    class FakeBrowser:
        def __init__(self) -> None:
            self.calls = 0

        async def receive_json(self) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.01)
            return {"type": "finish"}

    class FakeSession:
        task_id = "task-1"

        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, message: dict[str, object]) -> None:
            self.messages.append(message)

    browser = FakeBrowser()
    session = FakeSession()

    await routes._wait_for_control(
        browser,
        session,
        asyncio.Event(),
        {"text_sent_at": None},
    )

    assert session.messages == [build_finish_task("task-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_flush", [False, True])
async def test_tts_task_rotates_provider_without_notifying_client(
    monkeypatch: pytest.MonkeyPatch,
    explicit_flush: bool,
) -> None:
    import fish_memory.tts.routes as routes

    monkeypatch.setattr(routes, "TTS_IDLE_ROTATION_INTERVAL_SECONDS", 0.001)

    class FakeBrowser:
        def __init__(self) -> None:
            self.calls = 0
            self.events: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                if explicit_flush:
                    return {"type": "flush"}
                await asyncio.sleep(0.01)
            if self.calls == 2:
                return {"type": "append", "text": "第二句。"}
            return {"type": "finish"}

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

        async def send_bytes(self, _: bytes) -> None:
            raise AssertionError("the fake provider emits no audio")

    class RotatingProvider:
        def __init__(self) -> None:
            self.task_id = ""
            self.task_count = 0
            self.messages: list[dict[str, object]] = []
            self.waiting_for_started = False
            self.finish_requested = asyncio.Event()

        def begin_task(self) -> str:
            self.task_count += 1
            self.task_id = f"task-{self.task_count}"
            return self.task_id

        async def send(self, message: dict[str, object]) -> None:
            self.messages.append(message)
            action = message.get("header", {}).get("action")
            if action == "run-task":
                self.waiting_for_started = True
            elif action == "finish-task":
                self.finish_requested.set()

        async def receive(self) -> str:
            if self.waiting_for_started:
                self.waiting_for_started = False
                return json.dumps({"header": {"event": "task-started"}})
            await self.finish_requested.wait()
            self.finish_requested.clear()
            return json.dumps({"header": {"event": "task-finished"}})

    browser = FakeBrowser()
    provider = RotatingProvider()
    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        audio_format="pcm",
        sample_rate=22_050,
    )

    provider_reusable = await routes._run_tts_task(
        browser,
        provider,
        request,
        started_at=0.0,
        connection_reused=True,
        connection_ms=None,
    )

    assert provider_reusable is True
    assert provider.task_count == 2
    assert browser.calls == 3
    assert [event["type"] for event in browser.events] == [
        "ready",
        "segment_finished",
        "finished",
    ]
    continue_messages = [
        message
        for message in provider.messages
        if message["header"]["action"] == "continue-task"
    ]
    assert len(continue_messages) == 1
    assert continue_messages[0]["payload"] == {"input": {"text": "第二句。"}}


@pytest.mark.asyncio
async def test_cancelled_task_keeps_client_socket_when_provider_disconnects() -> None:
    from fish_memory.tts.routes import _run_tts_task

    class FakeBrowser:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            return {"type": "stop"}

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    class DisconnectingProvider:
        task_id = "task-1"

        def __init__(self) -> None:
            self.cancelled = asyncio.Event()
            self.receive_count = 0

        def begin_task(self) -> str:
            return self.task_id

        async def send(self, message: dict[str, object]) -> None:
            payload = message.get("payload")
            if isinstance(payload, dict):
                input_payload = payload.get("input")
                if (
                    isinstance(input_payload, dict)
                    and input_payload.get("directive") == "cancel"
                ):
                    self.cancelled.set()

        async def receive(self) -> str:
            self.receive_count += 1
            if self.receive_count == 1:
                return json.dumps({"header": {"event": "task-started"}})
            await self.cancelled.wait()
            raise RuntimeError("provider disconnected after cancellation")

    browser = FakeBrowser()
    provider = DisconnectingProvider()
    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        audio_format="pcm",
        sample_rate=22_050,
    )

    provider_reusable = await _run_tts_task(
        browser,
        provider,
        request,
        started_at=0.0,
        connection_reused=True,
        connection_ms=None,
    )

    assert provider_reusable is False
    assert browser.events[-1]["type"] == "finished"
    assert browser.events[-1]["stopped"] is True


@pytest.mark.asyncio
async def test_cancelled_task_times_out_without_closing_client_socket() -> None:
    from fish_memory.tts.routes import _run_tts_task

    class FakeBrowser:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            return {"type": "stop"}

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

    class HangingProvider:
        task_id = "task-1"

        def __init__(self) -> None:
            self.receive_count = 0

        def begin_task(self) -> str:
            return self.task_id

        async def send(self, _: dict[str, object]) -> None:
            return None

        async def receive(self) -> str:
            self.receive_count += 1
            if self.receive_count == 1:
                return json.dumps({"header": {"event": "task-started"}})
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    browser = FakeBrowser()
    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        audio_format="pcm",
        sample_rate=22_050,
    )

    provider_reusable = await asyncio.wait_for(
        _run_tts_task(
            browser,
            HangingProvider(),
            request,
            started_at=0.0,
            connection_reused=True,
            connection_ms=None,
        ),
        timeout=1.2,
    )

    assert provider_reusable is False
    assert browser.events[-1]["type"] == "finished"
    assert browser.events[-1]["stopped"] is True


@pytest.mark.asyncio
async def test_cancelled_task_does_not_raise_after_client_disconnects() -> None:
    from fish_memory.tts.routes import _run_tts_task

    class DisconnectedBrowser:
        def __init__(self) -> None:
            self.disconnected = False
            self.events: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            self.disconnected = True
            raise WebSocketDisconnect()

        async def send_json(self, event: dict[str, object]) -> None:
            if self.disconnected:
                raise WebSocketDisconnect()
            self.events.append(event)

    class CancelledProvider:
        task_id = "task-1"

        def __init__(self) -> None:
            self.cancelled = asyncio.Event()
            self.receive_count = 0

        def begin_task(self) -> str:
            return self.task_id

        async def send(self, message: dict[str, object]) -> None:
            payload = message.get("payload")
            if isinstance(payload, dict):
                input_payload = payload.get("input")
                if (
                    isinstance(input_payload, dict)
                    and input_payload.get("directive") == "cancel"
                ):
                    self.cancelled.set()

        async def receive(self) -> str:
            self.receive_count += 1
            if self.receive_count == 1:
                return json.dumps({"header": {"event": "task-started"}})
            await self.cancelled.wait()
            raise RuntimeError("provider task cancelled")

    browser = DisconnectedBrowser()
    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        audio_format="pcm",
        sample_rate=22_050,
    )

    provider_reusable = await _run_tts_task(
        browser,
        CancelledProvider(),
        request,
        started_at=0.0,
        connection_reused=True,
        connection_ms=None,
    )

    assert provider_reusable is False
    assert [event["type"] for event in browser.events] == ["ready"]


@pytest.mark.asyncio
async def test_reused_client_socket_picks_up_new_tts_credentials_between_tasks(
    monkeypatch,
):
    from types import SimpleNamespace

    from pydantic import SecretStr

    import fish_memory.tts.routes as routes
    from fish_memory.config import Settings

    settings = Settings.model_construct(
        tts_provider="dashscope", tts_api_key=SecretStr("old-key")
    )
    sessions = []

    class Session:
        def __init__(self, *, api_key, websocket_url):
            self.key = api_key
            self.is_connected = False
            self.closed = 0
            sessions.append(self)

        async def connect(self):
            self.is_connected = True

        async def close(self):
            self.is_connected = False
            self.closed += 1

    completed = []

    async def run(websocket, session, request, **kwargs):
        completed.append(session.key)
        if len(completed) == 1:
            settings.tts_api_key = SecretStr("new-key")
        # A running task still owns its original connection after saving.
        assert session.key == completed[-1]
        return True

    class Socket:
        app = SimpleNamespace(state=SimpleNamespace(settings=settings))
        events = []

        async def accept(self):
            pass

        async def receive_json(self):
            if len(completed) >= 3:
                raise WebSocketDisconnect()
            return {
                "type": "start",
                "model": "qwen-audio-3.0-tts-flash",
                "voice": "longpaopao_v3.6",
                "format": "pcm",
                "sampleRate": 16000,
            }

        async def send_json(self, event):
            self.events.append(event)

        async def close(self, code=1000):
            pass

    monkeypatch.setattr(routes, "DashScopeTtsSession", Session)
    monkeypatch.setattr(routes, "_run_tts_task", run)
    socket = Socket()
    await routes.tts_stream(socket)
    assert completed == ["old-key", "new-key", "new-key"]
    assert len(sessions) == 2
    assert [session.closed for session in sessions] == [1, 1]
    assert [event["connectionReused"] for event in socket.events] == [
        False,
        False,
        True,
    ]


def test_failed_speech_keeps_the_provider_error_code() -> None:
    from fish_memory import failures

    failed = parse_server_event(
        {
            "header": {
                "event": "task-failed",
                "error_code": "InvalidParameter",
                "error_message": "Please ensure input text is valid.",
            },
            "payload": {},
        }
    )
    assert failed.kind == "failed"
    assert failures.kind_of(failed.failure()) == "rejected"


@pytest.mark.asyncio
async def test_speech_failure_tells_the_companion_why(monkeypatch):
    from types import SimpleNamespace

    from pydantic import SecretStr

    import fish_memory.tts.routes as routes
    from fish_memory import failures
    from fish_memory.config import Settings

    health = failures.ServiceHealth()
    monkeypatch.setattr(failures, "health", health)
    settings = Settings.model_construct(
        tts_provider="dashscope", tts_api_key=SecretStr("test-only")
    )

    class Session:
        def __init__(self, *, api_key, websocket_url):
            self.is_connected = False

        async def connect(self):
            self.is_connected = True

        async def close(self):
            self.is_connected = False

    async def run(websocket, session, request, **kwargs):
        raise failures.ProviderError("Arrearage", "Arrearage")

    class Socket:
        app = SimpleNamespace(state=SimpleNamespace(settings=settings))
        events: list[dict[str, object]] = []

        async def accept(self):
            pass

        async def receive_json(self):
            return {
                "type": "start",
                "model": "qwen-audio-3.0-tts-flash",
                "voice": "longpaopao_v3.6",
                "format": "pcm",
                "sampleRate": 16000,
            }

        async def send_json(self, event):
            self.events.append(event)

        async def close(self, code=1000):
            pass

    monkeypatch.setattr(routes, "DashScopeTtsSession", Session)
    monkeypatch.setattr(routes, "_run_tts_task", run)
    socket = Socket()
    await routes.tts_stream(socket)
    assert socket.events[-1] == {
        "type": "error",
        "message": "Arrearage",
        "failure": "voice:quota",
    }
    assert health.problems() == (failures.ServiceFailure("voice", "quota"),)
