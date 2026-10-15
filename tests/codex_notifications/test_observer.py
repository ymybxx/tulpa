import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fish_memory.codex_notifications.client import CodexReadClient
from fish_memory.codex_notifications.observer import CompletionObserver, completion


def thread():
    return {
        "id": "thread-1",
        "name": "检查构建",
        "cwd": "/work/project",
        "updatedAt": 105,
        "status": {"type": "notLoaded"},
    }


def turn(status="completed", ended=105, text="构建检查通过。"):
    return {
        "id": "turn-1",
        "status": status,
        "completedAt": ended,
        "error": None,
        "items": [{"type": "agentMessage", "phase": "final_answer", "text": text}],
    }


class Hints:
    def __init__(self):
        self.scanned = asyncio.Event()
        self.hint = asyncio.Event()

    def watch(self, files, directory):
        self.scanned.set()

    def remember(self, thread_id, turn_id):
        pass

    async def wait(self):
        await self.hint.wait()
        self.hint.clear()
        return {"thread-1"}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def isolated_watches(monkeypatch):
    monkeypatch.setattr(
        "fish_memory.codex_notifications.observer.CompletionHints", Hints
    )


class Client:
    def __init__(self):
        self.turns = [turn()]
        self.calls = []
        self.fail = False

    async def read(self, method, params):
        self.calls.append((method, params))
        if self.fail:
            raise EOFError()
        if method == "thread/read":
            return {"thread": thread()}
        return {
            "data": [thread()] if method == "thread/list" else self.turns,
            "nextCursor": None,
        }

    async def close(self):
        pass


@pytest.mark.parametrize(
    "value",
    [
        turn("interrupted"),
        turn("inProgress"),
        turn(ended=None),
        turn(ended=99),
        turn(text='{"disposition":"input_required"}'),
    ],
)
def test_requires_explicit_final_completion(value):
    assert completion(thread(), value, 100) is None


def test_final_answer_and_main_thread_only():
    value = turn()
    value["items"][0]["phase"] = "commentary"
    assert completion(thread(), value, 100) is None
    assert completion({**thread(), "parentThreadId": "parent"}, turn(), 100) is None
    assert completion(thread(), turn(), 100)["title"] == "检查构建"


async def test_durable_receipt_restart_and_no_history_replay(tmp_path):
    clock = [100]
    client = Client()
    client.turns = [turn(ended=99)]
    deliver = AsyncMock(return_value=True)
    path = tmp_path / "state.json"
    observer = CompletionObserver(
        client, path, deliver, enabled=True, now=lambda: clock[0]
    )
    await observer.poll()
    deliver.assert_not_called()
    clock[0] = 106
    client.turns = [turn()]
    await observer.poll()
    assert deliver.await_count == 1
    observer = CompletionObserver(
        client, path, deliver, enabled=True, now=lambda: clock[0]
    )
    await observer.poll()
    assert deliver.await_count == 1
    assert {m for m, _ in client.calls} == {"thread/list", "thread/turns/list"}


async def test_restart_discards_notifications_pending_from_previous_run(
    tmp_path, monkeypatch
):
    read_text = Path.read_text

    def default_gbk(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "gbk", errors=errors)

    monkeypatch.setattr(Path, "read_text", default_gbk)
    path = tmp_path / "state.json"
    client = Client()
    deliver = AsyncMock(return_value=False)
    observer = CompletionObserver(client, path, deliver, enabled=True, now=lambda: 100)
    await observer.poll()
    assert len(json.loads(path.read_text(encoding="utf-8"))["pending"]) == 1
    deliver.return_value = True
    observer = CompletionObserver(client, path, deliver, enabled=True, now=lambda: 120)
    await observer.poll()
    assert not json.loads(path.read_text(encoding="utf-8"))["pending"]
    assert deliver.await_count == 1
    assert json.loads(path.read_text(encoding="utf-8"))["since"] == 120


async def test_restart_skips_offline_completions_but_announces_new_ones(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "since": 10,
                "cursor": 90,
                "seen": ["already-delivered"],
                "pending": {},
            }
        )
    )
    clock = [200]
    metadata = {**thread(), "updatedAt": 201}
    monkeypatch.setattr(__name__ + ".thread", lambda: metadata)
    client = Client()
    client.turns = [turn(ended=190)]
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, path, deliver, enabled=True, now=lambda: clock[0]
    )
    await observer.poll()
    deliver.assert_not_called()
    assert observer.state["since"] == 200
    assert observer.state["seen"] == ["already-delivered"]

    clock[0] = 206
    client.turns.insert(0, {**turn(ended=205), "id": "new-turn"})
    await observer.poll()
    await observer.poll()
    deliver.assert_awaited_once()
    assert deliver.call_args.args[0]["turn_id"] == "new-turn"


async def test_delayed_first_scan_keeps_completions_after_enable_time(tmp_path):
    clock = [100]
    client = Client()
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, tmp_path / "state", deliver, now=lambda: clock[0]
    )
    observer.configure(True)
    clock[0] = 110
    await observer.poll()
    deliver.assert_awaited_once()
    assert observer.state["since"] == 100


async def test_connection_failure_does_not_reset_running_session_baseline(tmp_path):
    clock = [100]
    client = Client()
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, tmp_path / "state", deliver, enabled=True, now=lambda: clock[0]
    )
    client.fail = True
    with pytest.raises(EOFError):
        await observer.poll()
    clock[0] = 110
    client.fail = False
    await observer.poll()
    assert observer.state["since"] == 100
    deliver.assert_awaited_once()


async def test_disable_no_io_and_reenable_new_baseline(tmp_path):
    clock = [100]
    client = Client()
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(client, tmp_path / "s", deliver, now=lambda: clock[0])
    await observer.poll()
    assert not client.calls
    observer.configure(True)
    await observer.poll()
    observer.configure(False)
    clock[0] = 200
    await observer.poll()
    observer.configure(True)
    await observer.poll()
    assert deliver.await_count == 1


async def test_failure_does_not_advance_checkpoint(tmp_path):
    client = Client()
    observer = CompletionObserver(
        client, tmp_path / "s", AsyncMock(), enabled=True, now=lambda: 100
    )
    client.fail = True
    with pytest.raises(EOFError):
        await observer.poll()
    assert json.loads((tmp_path / "s").read_text(encoding="utf-8"))["cursor"] == 100


async def test_client_cannot_resume_execute_or_approve():
    client = CodexReadClient()
    for method in (
        "thread/resume",
        "turn/start",
        "turn/interrupt",
        "account/login/start",
    ):
        with pytest.raises(ValueError):
            await client.read(method, {})
    assert client.process is None


async def test_unchanged_rollouts_do_not_reload_turns(tmp_path, monkeypatch):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("before")
    metadata = {**thread(), "path": str(rollout)}
    monkeypatch.setattr(__name__ + ".thread", lambda: metadata)
    client = Client()
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, tmp_path / "s", deliver, enabled=True, now=lambda: 100
    )
    for _ in range(12):
        await observer.poll()
    assert sum(m == "thread/list" for m, _ in client.calls) == 12
    assert sum(m == "thread/turns/list" for m, _ in client.calls) == 1
    assert deliver.await_count == 1
    # updatedAt has only second precision; a file append must invalidate the cache.
    rollout.write_text("after, same metadata timestamp")
    client.turns = [{**turn(), "id": "turn-2"}]
    await observer.poll()
    assert sum(m == "thread/turns/list" for m, _ in client.calls) == 2
    assert deliver.await_count == 2


async def test_pending_results_drain_without_waiting_for_another_scan(tmp_path):
    client = Client()
    client.turns.append({**turn(), "id": "turn-2"})
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, tmp_path / "s", deliver, enabled=True, now=lambda: 100
    )
    await observer.poll()
    assert deliver.await_count == 2
    assert not observer.state["pending"]


async def test_active_turn_still_refreshes_when_api_settles_after_file_flush(
    tmp_path, monkeypatch
):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("already flushed")
    metadata = {**thread(), "path": str(rollout)}
    monkeypatch.setattr(__name__ + ".thread", lambda: metadata)
    client = Client()
    client.turns = [turn("inProgress", ended=None)]
    deliver = AsyncMock(return_value=True)
    observer = CompletionObserver(
        client, tmp_path / "s", deliver, enabled=True, now=lambda: 100
    )
    await observer.poll()
    client.turns = [turn()]
    await observer.poll()
    deliver.assert_awaited_once()


async def test_completion_hint_wakes_without_increasing_idle_poll_frequency(tmp_path):
    client = Client()
    client.turns = []
    delivered = asyncio.Event()

    async def deliver(event, key):
        delivered.set()
        return True

    observer = CompletionObserver(
        client, tmp_path / "s", deliver, enabled=True, now=lambda: 100, interval=60
    )
    observer.start()
    try:
        await asyncio.wait_for(observer._hints.scanned.wait(), 1)
        client.turns = [turn()]
        observer._hints.hint.set()
        await asyncio.wait_for(delivered.wait(), 1)
        scans = sum(m == "thread/list" for m, _ in client.calls)
        assert scans == 1
        assert sum(m == "thread/read" for m, _ in client.calls) == 1
        await asyncio.sleep(0.1)
        assert sum(m == "thread/list" for m, _ in client.calls) == scans
    finally:
        await observer.close()
