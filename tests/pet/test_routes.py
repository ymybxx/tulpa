from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.routes import router
from fish_memory.runtime_configuration import runtime_configuration_router


def test_quick_chat_uses_authenticated_native_command_and_waits_for_acceptance(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    headers = {"Authorization": "Bearer fm_pet_secret"}
    body = {
        "requestId": str(uuid4()),
        "fishId": str(uuid4()),
        "content": "local fixture",
    }
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        assert client.post("/v1/pet/chat", json=body).status_code == 401
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as publisher:
            request = executor.submit(
                client.post, "/v1/pet/chat", json=body, headers=headers
            )
            command = publisher.receive_json()
            assert command["type"] == "pet.chat.submit"
            assert command["payload"] == {**body, "attachmentIds": []}
            assert not request.done()
            reply = {
                "requestId": body["requestId"],
                "ok": True,
                "messageId": str(uuid4()),
            }
            publisher.send_json({"type": "pet.chat.result", "payload": reply})
            assert request.result(timeout=3).json() == reply
            assert (
                client.post("/v1/pet/chat", json=body, headers=headers).json() == reply
            )
            assert not app.state.pet_event_hub.snapshot()["payload"]["events"]


def build_app(tmp_path: Path) -> FastAPI:
    credentials = ApiKeyCredentialStore(
        api_key="fm_pet_secret",
        key_file=tmp_path / "api-key",
    )
    auth = ApiAuthService(credentials)
    app = FastAPI()
    app.state.pet_event_hub = PetEventHub()
    app.add_middleware(ApiAuthMiddleware, auth_service=auth)
    app.include_router(router)
    return app


def test_pet_websockets_are_authenticated_and_relay_control(tmp_path: Path) -> None:
    app = build_app(tmp_path)
    headers = {"Authorization": "Bearer fm_pet_secret"}

    with TestClient(app) as client:
        with client.websocket_connect("/v1/pet/events", headers=headers) as subscriber:
            assert subscriber.receive_json()["type"] == "pet.snapshot"
            with client.websocket_connect(
                "/v1/pet/events/publish",
                headers=headers,
            ) as publisher:
                publisher.send_json(
                    {
                        "type": "pet.state",
                        "generation": None,
                        "payload": {"state": "standby"},
                    }
                )
                event = subscriber.receive_json()

    assert event["type"] == "pet.state"
    assert event["payload"] == {"state": "standby"}


def test_pet_websocket_rejects_anonymous_client(tmp_path: Path) -> None:
    with (
        pytest.raises(Exception) as exc_info,
        TestClient(build_app(tmp_path)) as client,
    ):
        with client.websocket_connect("/v1/pet/events"):
            pass

    assert getattr(exc_info.value, "code", None) == 4401


def test_plugin_playback_sessions_reach_native_and_replay_on_reconnect(tmp_path):
    app = build_app(tmp_path)
    hub = app.state.pet_event_hub
    headers = {"Authorization": "Bearer fm_pet_secret"}
    bindings = {
        "sessions": [
            {
                "fishId": "00000000-0000-0000-0000-000000000001",
                "sessionId": "00000000-0000-0000-0000-000000000201",
            }
        ]
    }

    @app.post("/test-plugin-sessions")
    async def publish(payload: dict):
        hub.publish_control("pet.plugin.sessions", None, payload)
        return {"ok": True}

    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as native:
            assert (
                client.post(
                    "/test-plugin-sessions", json=bindings, headers=headers
                ).status_code
                == 200
            )
            assert native.receive_json()["payload"] == bindings
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as native:
            assert native.receive_json()["payload"] == bindings
            client.post("/test-plugin-sessions", json={"sessions": []}, headers=headers)
            assert native.receive_json()["payload"] == {"sessions": []}
    assert hub.publisher_count == 0
    assert hub.subscriber_count == 0


def test_audio_controls_are_independent_authenticated_and_replayed(tmp_path):
    app = build_app(tmp_path)
    app.state.settings = SimpleNamespace(desktop_mode=True)
    app.include_router(runtime_configuration_router)
    headers = {"Authorization": "Bearer fm_pet_secret"}
    with TestClient(app) as client:
        assert (
            client.put(
                "/internal/runtime/microphone", json={"enabled": False}
            ).status_code
            == 401
        )
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as native:
            for route, event in (
                ("microphone", "pet.microphone"),
                ("playback", "pet.speaker"),
            ):
                assert (
                    client.put(
                        f"/internal/runtime/{route}",
                        json={"enabled": False},
                        headers=headers,
                    ).status_code
                    == 200
                )
                value = native.receive_json()
                assert (value["type"], value["payload"]) == (event, {"enabled": False})
            assert (
                client.put(
                    "/internal/runtime/microphone",
                    json={"enabled": "false"},
                    headers=headers,
                ).status_code
                == 422
            )
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as native:
            replay = [native.receive_json(), native.receive_json()]
            assert {item["type"] for item in replay} == {
                "pet.microphone",
                "pet.speaker",
            }
            assert all(item["payload"] == {"enabled": False} for item in replay)


def test_speech_facial_profile_is_authenticated_validated_and_clearable(tmp_path):
    from types import SimpleNamespace

    from fish_memory.experimental.config import ExperimentalConfiguration

    app = build_app(tmp_path)
    app.state.experimental_companion = SimpleNamespace(
        configuration=ExperimentalConfiguration(api_key="fixture")
    )
    headers = {"Authorization": "Bearer fm_pet_secret"}
    profile = {
        "revision": "v1",
        "modelId": "cat",
        "modelName": "cat",
        "parameters": [
            {"id": "ParamAngleX", "minimum": -30, "maximum": 30, "defaultValue": 0}
        ],
    }
    with TestClient(app) as client:
        assert client.put("/v1/pet/facial-profile", json=profile).status_code == 401
        response = client.put("/v1/pet/facial-profile", json=profile, headers=headers)
        assert response.status_code == 200 and response.json() == {"enabled": True}
        binary = {
            **profile,
            "parameters": [
                {
                    "id": "ParamCheek",
                    "minimum": 0,
                    "maximum": 1,
                    "defaultValue": 0,
                    "mode": "binary",
                    "falseDescription": "不显示红晕",
                    "trueDescription": "显示红晕",
                }
            ],
        }
        assert client.put(
            "/v1/pet/facial-profile", json=binary, headers=headers
        ).json() == {"enabled": True}
        old = app.state.speech_facial_profile
        invalid = {
            **profile,
            "parameters": [
                {"id": "ParamBreath", "minimum": 0, "maximum": 1, "defaultValue": 0}
            ],
        }
        assert (
            client.put(
                "/v1/pet/facial-profile", json=invalid, headers=headers
            ).status_code
            == 422
        )
        assert app.state.speech_facial_profile is old
        assert client.put(
            "/v1/pet/facial-profile", json=None, headers=headers
        ).json() == {"enabled": False}
        assert app.state.speech_facial_profile is None


def test_listen_reaches_native_only_when_it_is_connected(tmp_path: Path) -> None:
    app = build_app(tmp_path)
    headers = {"Authorization": "Bearer fm_pet_secret"}
    with TestClient(app) as client:
        assert client.post("/v1/pet/listen").status_code == 401
        assert client.post("/v1/pet/listen", headers=headers).status_code == 503
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as publisher:
            assert client.post("/v1/pet/listen", headers=headers).json() == {"ok": True}
            assert publisher.receive_json()["type"] == "pet.listen"
            stop = client.post("/v1/pet/listen/stop", headers=headers)
            assert stop.json() == {"ok": True}
            assert publisher.receive_json()["type"] == "pet.listen.stop"


def test_new_conversation_creates_a_session_and_tells_native(tmp_path: Path) -> None:
    from dataclasses import dataclass
    from uuid import UUID, uuid4

    from fish_memory.db.sessions import SessionNotFoundError

    fish = uuid4()
    created: list[UUID] = []

    @dataclass
    class Session:
        id: UUID

    class Store:
        async def create(self, fish_id: UUID) -> Session:
            if fish_id != fish:
                raise SessionNotFoundError(fish_id)
            created.append(uuid4())
            return Session(created[-1])

    class History:
        active: list[tuple[UUID, UUID]] = []

        def set_active(self, fish_id: UUID, session_id: UUID) -> None:
            self.active.append((fish_id, session_id))

    app = build_app(tmp_path)
    app.state.session_store = Store()
    app.state.session_history = History()
    headers = {"Authorization": "Bearer fm_pet_secret"}
    body = {"fishId": str(fish)}
    with TestClient(app) as client:
        assert client.post("/v1/pet/session/new", json=body).status_code == 401
        missing = {"fishId": str(uuid4())}
        assert (
            client.post("/v1/pet/session/new", json=missing, headers=headers)
        ).status_code == 404
        # Typed chat with voice off still lands in the new session.
        alone = client.post("/v1/pet/session/new", json=body, headers=headers)
        assert alone.json() == {"sessionId": str(created[-1])}
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as publisher:
            reply = client.post("/v1/pet/session/new", json=body, headers=headers)
            assert reply.json() == {"sessionId": str(created[-1])}
            event = publisher.receive_json()
            assert event["type"] == "pet.session.new"
            assert event["payload"] == {
                "fishId": str(fish),
                "sessionId": str(created[-1]),
            }
        assert app.state.session_history.active[-1] == (fish, created[-1])


def test_adding_missed_owner_speech_reaches_native_only_when_connected(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    headers = {"Authorization": "Bearer fm_pet_secret"}
    with TestClient(app) as client:
        assert client.post("/v1/pet/owner-voice/add").status_code == 401
        assert (
            client.post("/v1/pet/owner-voice/add", headers=headers).status_code == 503
        )
        with client.websocket_connect(
            "/v1/pet/events/publish", headers=headers
        ) as publisher:
            response = client.post("/v1/pet/owner-voice/add", headers=headers)
            assert response.json() == {"ok": True}
            assert publisher.receive_json()["type"] == "pet.owner_voice.add"
