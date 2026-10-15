import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fish_memory.extensions.manager import PluginManager
from fish_memory.extensions.models import CreateInstance, ExtensionError
from fish_memory.extensions.routes import extension_router
from fish_memory.runtime_platform import private_file_is_private
from tests.extensions.test_service import connected, make_service


def notice(text):
    return {
        "event_id": str(uuid4()),
        "type": "example.notice",
        "data": {"text": text},
    }


def test_saved_plugin_binding_is_ignored_and_no_longer_persisted(tmp_path, monkeypatch):
    identifier = uuid4()
    index = tmp_path / "installed.json"
    index.write_text(
        json.dumps(
            [
                {
                    "id": str(identifier),
                    "manifest": {
                        "id": "test.live",
                        "name": "中文插件 🐟",
                        "version": "1.0.0",
                        "entrypoint": "test_plugin:Plugin",
                    },
                    "path": str(tmp_path),
                    "python": "/test/python",
                    "development": False,
                    "description": {},
                    "fish_id": str(uuid4()),
                    "session_id": str(uuid4()),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_read = Path.read_text
    original_write = Path.write_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding=None, **kwargs: original_read(
            self, encoding=encoding or "gbk", **kwargs
        ),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, data, encoding=None, **kwargs: original_write(
            self, data, encoding=encoding or "gbk", **kwargs
        ),
    )
    manager = PluginManager(SimpleNamespace(), SimpleNamespace(), tmp_path)
    item = manager.items[identifier]
    assert item.fish_id is None and item.session_id is None
    item.fish_id, item.session_id = uuid4(), uuid4()
    manager._save()
    saved = json.loads(index.read_text(encoding="utf-8"))[0]
    assert saved["id"] == str(identifier)
    assert saved["manifest"]["name"] == "中文插件 🐟"
    assert "fish_id" not in saved and "session_id" not in saved


@pytest.mark.asyncio
async def test_development_token_is_private_before_writing(tmp_path, monkeypatch):
    import fish_memory.extensions.manager as module

    manager = PluginManager(SimpleNamespace(), SimpleNamespace(), tmp_path / "plugins")
    original_protect = module.private_file_permissions
    protected = []

    def protect_empty(path, descriptor):
        assert path.stat().st_size == 0
        original_protect(path, descriptor)
        protected.append(path)

    monkeypatch.setattr(module, "private_file_permissions", protect_empty)
    await manager.development(True, "http://127.0.0.1:8000")
    discovery = tmp_path / "plugin-dev.json"
    assert protected == [discovery]
    assert private_file_is_private(discovery)
    payload = json.loads(discovery.read_text(encoding="utf-8"))
    assert payload["token"] == manager._dev_token
    assert payload["token"]
    await manager.development(False, "http://127.0.0.1:8000")
    assert manager._dev_token is None
    assert not discovery.exists()


@pytest.mark.asyncio
async def test_development_permission_failure_leaves_no_token(tmp_path, monkeypatch):
    import fish_memory.extensions.manager as module

    manager = PluginManager(SimpleNamespace(), SimpleNamespace(), tmp_path / "plugins")

    def fail_protection(path, descriptor):
        assert path.stat().st_size == 0
        raise PermissionError("ACL denied")

    monkeypatch.setattr(module, "private_file_permissions", fail_protection)
    with pytest.raises(PermissionError, match="ACL denied"):
        await manager.development(True, "http://127.0.0.1:8000")
    assert manager._dev_token is None
    assert not (tmp_path / "plugin-dev.json").exists()


@pytest.mark.asyncio
async def test_start_uses_current_conversation_even_with_stale_explicit_binding():
    service = make_service()
    target = uuid4(), uuid4()
    service.session_target = AsyncMock(return_value=target)
    try:
        item, _ = await service.create(
            CreateInstance(plugin_id="test.live", fish_id=uuid4(), session_id=uuid4())
        )
        assert (item.fish_id, item.session_id) == target
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_connected_plugin_follows_session_and_pet_without_reconnect():
    service = make_service()
    item, token = await connected(service, speak=False, autonomous=False)
    original = item.fish_id, item.session_id
    generation, definitions = item.generation, item.definitions
    service.session_target = AsyncMock(return_value=original)
    try:
        for target in ((original[0], uuid4()), (uuid4(), uuid4())):
            service.session_target.return_value = target
            service.native_consumer(*target, True)
            await asyncio.gather(*tuple(service._background))
            assert (item.fish_id, item.session_id) == target
            assert item.connected and not item.stopped
            assert item.generation == generation and item.definitions is definitions
            assert (
                service.authenticate([(b"authorization", f"Bearer {token}".encode())])
                is item
            )
            await service.request(item, "context", {"fields": ["history"]})
            service.contexts.read.assert_awaited_with(*target, ["history"])
            receipt = await service.request(item, "push", notice("新会话的弹幕"))
            assert receipt["status"] == "stored_only"
        # The plugin must not instruct the companion to adopt its old session.
        leases = [
            e["payload"]["sessions"]
            for e in service.hub.snapshot()["payload"]["events"]
            if e["type"] == "pet.plugin.sessions"
        ]
        assert leases == [[]]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_inflight_receipt_keeps_its_session_and_next_event_uses_new_one():
    service = make_service()
    item, _ = await connected(service, speak=False, autonomous=False)
    original = item.fish_id, item.session_id
    target = original[0], uuid4()
    service.session_target = AsyncMock(return_value=original)
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []
    receive = service.store.receive

    async def delayed(*args):
        entered.set()
        await release.wait()
        writes.append((args[0].fish_id, args[0].session_id))
        return await receive(*args)

    service.store.receive = delayed
    try:
        first = asyncio.create_task(service.request(item, "push", notice("旧会话")))
        await entered.wait()
        service.session_target.return_value = target
        switching = asyncio.create_task(service.sync_session())
        await asyncio.sleep(0)
        assert (item.fish_id, item.session_id) == original
        release.set()
        await first
        await switching
        await service.request(item, "push", notice("新会话"))
        assert writes == [original, target]
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
async def test_session_switch_does_not_replay_old_queued_speech():
    service = make_service()
    item, _ = await connected(service, autonomous=False)
    service.session_target = AsyncMock(return_value=(item.fish_id, item.session_id))
    service.foreground = lambda: True
    try:
        event = notice("旧会话中等待播放的内容")
        assert (await service.request(item, "push", event))["status"] == "queued"
        service.session_target.return_value = item.fish_id, uuid4()
        await service.sync_session()
        receipt = await service.request(item, "result", {"event_id": event["event_id"]})
        assert receipt["status"] == "interrupted"
        assert receipt["reason"] == "session_changed"
        assert not service._pending_events and not service.llm.calls
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_missing_current_conversation_never_writes_to_stale_session():
    service = make_service()
    item, _ = await connected(service, speak=False, autonomous=False)
    service.session_target = AsyncMock(return_value=None)
    try:
        await service.sync_session()
        with pytest.raises(ExtensionError, match="当前没有可用"):
            await service.request(item, "push", notice("无当前会话"))
        assert not service.store.values
    finally:
        await service.close()


@pytest.mark.parametrize("stale_binding", [False, True])
def test_plugin_management_no_longer_accepts_a_separate_session_selection(
    stale_binding,
):
    app = FastAPI()
    app.include_router(extension_router)
    manager = SimpleNamespace(
        start=AsyncMock(
            return_value=SimpleNamespace(public=lambda: {"status": "starting"})
        ),
        development=AsyncMock(),
    )
    app.state.plugin_manager = manager
    identifier = uuid4()
    body = {"id": str(identifier), "config": {"mode": "mock"}}
    if stale_binding:
        body.update(fish_id=str(uuid4()), session_id=str(uuid4()))
    with TestClient(app) as client:
        response = client.post(
            "/internal/extensions/manager/start",
            json=body,
        )
        assert response.status_code == 200
        manager.start.assert_awaited_once_with(
            identifier, {"mode": "mock"}, "http://testserver"
        )
        response = client.post(
            "/internal/extensions/manager/development", json={"enabled": True}
        )
        assert response.status_code == 200
        manager.development.assert_awaited_once_with(True, "http://testserver")
