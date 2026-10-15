import asyncio
import hashlib
import json
import zipfile

import pytest
from fish_plugin_sdk import Activity, Event, Input, Pet, Plugin
from fish_plugin_sdk.manifest import Bundle, Manifest, extract_bundle, target
from fish_plugin_sdk.runner import deliver_inputs
from pydantic import BaseModel
from websockets.asyncio.server import serve


class Notice(BaseModel):
    text: str


EVENT = Event("example.notice", Notice, "解释这件事", False)


def test_mergeability_is_explicit_and_legacy_events_keep_their_wire_shape():
    assert "mergeable" not in EVENT.definition()
    merged = Event("example.enter", Notice, "一起欢迎到场者", mergeable=True)
    assert merged.definition()["mergeable"] is True
    with pytest.raises(ValueError, match="mergeable"):
        Event("example.enter", Notice, "欢迎", mergeable="yes").definition()


@pytest.mark.asyncio
async def test_plain_text_reason_round_trip_keeps_empty_fields_and_legacy_mode():
    requests = []

    async def host(socket):
        async for raw in socket:
            request = json.loads(raw)
            if request["method"] == "reason":
                requests.append(request["params"])
                result = {"value": {"text": "结果"}}
            else:
                result = {}
            await socket.send(json.dumps({"id": request["id"], "result": result}))

    async with serve(host, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with Pet(f"ws://127.0.0.1:{port}", "test", []) as pet:
            result = await pet.reason(
                "判断话题", Notice, fields=(), input_text="话题名称"
            )
            assert result.text == "结果"
            await pet.reason("旧插件", Notice, data={"value": 1})
            with pytest.raises(ValueError):
                await pet.reason("错误", Notice, input_text="文字", data={"x": 1})
    assert len(requests) == 2
    assert requests[0]["input_text"] == "话题名称"
    assert requests[0]["fields"] == []
    assert requests[0]["schema"] == Notice.model_json_schema()
    assert "input_text" not in requests[1]
    assert requests[1]["data"] == {"value": 1}


@pytest.mark.asyncio
async def test_manual_input_round_trip_runs_plugin_before_event_push():
    seen = []
    received = asyncio.get_running_loop().create_future()

    class Example(Plugin):
        inputs = (Input("example.manual", Notice, "测试输入"),)
        events = (EVENT,)

        async def input(self, pet, type, data, *, request_id):
            seen.append((type, data.text, request_id))
            return await pet.push(EVENT, data, event_id=request_id)

    request_id = "2463789c-6f00-4223-bba6-080a64c3d187"

    async def host(socket):
        async for raw in socket:
            value = json.loads(raw)
            method, params = value["method"], value["params"]
            if method == "register":
                assert params["inputs"] == [Example.inputs[0].definition()]
                await socket.send(json.dumps({"id": value["id"], "result": {}}))
                await socket.send(
                    json.dumps(
                        {
                            "method": "input",
                            "params": {
                                "event_id": request_id,
                                "type": "example.manual",
                                "data": {"text": "hello"},
                            },
                        }
                    )
                )
            elif method == "push":
                assert seen == [("example.manual", "hello", request_id)]
                assert params["event_id"] == request_id
                await socket.send(
                    json.dumps(
                        {
                            "id": value["id"],
                            "result": {
                                "event_id": request_id,
                                "disposition": "stored_only",
                                "reason": "pet_busy",
                            },
                        }
                    )
                )
            elif method == "input_result":
                received.set_result(params)
                await socket.send(
                    json.dumps({"id": value["id"], "result": {"received": True}})
                )

    async with serve(host, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with Pet(
            f"ws://127.0.0.1:{port}",
            "test-token",
            [EVENT.definition()],
            inputs=[Example.inputs[0].definition()],
        ) as pet:
            worker = asyncio.create_task(deliver_inputs(Example({}), pet))
            try:
                async with asyncio.timeout(3):
                    result = await received
                assert result["request_id"] == request_id
                assert result["result"]["reason"] == "pet_busy"
                assert pet._notifications.empty()
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_task_capabilities_change_on_the_existing_connection():
    pet = Pet("ws://localhost", "test", [EVENT.definition()])
    pet.capabilities = {"tasks": False, "task_revision": 0}

    async def frames():
        yield json.dumps(
            {"method": "capabilities", "params": {"tasks": True, "task_revision": 1}}
        )

    await pet._read(frames())
    assert pet.tasks.available
    assert pet.capabilities["task_revision"] == 1
    assert pet._notifications.get_nowait()["method"] == "capabilities"


@pytest.mark.asyncio
async def test_reconnect_uncertain_push_reuses_id_and_scoped_context():
    received, ids = {}, []
    sockets = 0
    activity = Activity(name="直播", role="主播", audience="观众", objective="推进话题")
    registrations = []

    async def host(socket):
        nonlocal sockets
        sockets += 1
        async for raw in socket:
            request = json.loads(raw)
            method, params = request["method"], request["params"]
            if method == "register":
                registrations.append(params)
                await socket.send(
                    json.dumps({"id": request["id"], "result": {"registered": {}}})
                )
                await socket.send(
                    json.dumps({"method": "state", "params": {"can_respond": True}})
                )
            elif method == "push":
                ids.append(params["event_id"])
                if params["event_id"] not in received:
                    received[params["event_id"]] = params
                    await socket.close(1011, "lost ACK after committing")
                    return
                await socket.send(
                    json.dumps(
                        {
                            "id": request["id"],
                            "result": {"duplicate": True, "status": "stored_only"},
                        }
                    )
                )

    async with serve(host, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with Pet(
            f"ws://127.0.0.1:{port}",
            "scoped-token",
            [EVENT.definition()],
            activity=activity.model_dump(),
        ) as pet:
            result = await pet.push(EVENT, Notice(text="hello"))
            assert result["duplicate"] is True
            assert len(received) == 1 and len(ids) == 2 and ids[0] == ids[1]
        assert sockets == 2
        assert all(r["activity"] == activity.model_dump() for r in registrations)


def test_bundle_digest_and_path_checks_before_extract(tmp_path):
    manifest = Manifest(
        id="demo", name="Demo", version="0.1.0", entrypoint="demo:Plugin"
    )
    content = b"wheel bytes"
    bundle = Bundle(
        manifest=manifest,
        target=target(),
        files={"wheels/demo.whl": hashlib.sha256(content).hexdigest()},
    )
    path = tmp_path / "demo.fish-plugin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle.json", bundle.model_dump_json())
        archive.writestr("wheels/demo.whl", content + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        extract_bundle(path, tmp_path / "out", target())
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="Unsafe"):
        Bundle(manifest=manifest, target=target(), files={"../escape.whl": "0" * 64})
    with pytest.raises(ValueError, match="targets"):
        extract_bundle(path, tmp_path / "out", "unknown-platform")
