import asyncio
import json
import select
from uuid import uuid4

import pytest

from fish_memory.codex_notifications.changes import (
    CompletionHints,
    rollout_completion_hints,
)


def record(turn_id, *, kind="event_msg", event="task_complete"):
    return json.dumps({"type": kind, "payload": {"type": event, "turn_id": turn_id}})


def test_hints_require_complete_records_in_bounded_tails(tmp_path):
    thread_id, turn_id = str(uuid4()), str(uuid4())
    path = tmp_path / f"rollout-test-{thread_id}.jsonl"
    path.write_text(
        record(str(uuid4()))
        + "\n"
        + "x" * 100_000
        + "\n"
        + record(str(uuid4()), kind="response_item")
        + "\n"
        + record(str(uuid4()), event="task_started")
        + "\n"
        + record(turn_id)
        + "\n"
        + record(str(uuid4()))
    )
    assert rollout_completion_hints({path, tmp_path / "missing"}) == {
        (thread_id, turn_id)
    }


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="macOS/BSD vnode events")
async def test_completion_arrives_while_writer_remains_open_and_deduplicates(tmp_path):
    thread_id = str(uuid4())
    path = tmp_path / f"rollout-test-{thread_id}.jsonl"
    hints = CompletionHints()
    with path.open("ab") as writer:
        hints.watch({path}, tmp_path)
        try:
            writer.write(record(str(uuid4())).encode())
            writer.flush()
            waiter = asyncio.create_task(hints.wait())
            await asyncio.sleep(0.1)
            assert not waiter.done()  # A partial JSON line is not completion.
            writer.write(b"\n")
            writer.flush()
            assert await asyncio.wait_for(waiter, 1) == {thread_id}
            writer.write(b'{"type":"token_count"}\n')
            writer.flush()
            waiter = asyncio.create_task(hints.wait())
            await asyncio.sleep(0.1)
            assert not waiter.done()  # Same completed turn in the tail is deduped.
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        finally:
            hints.close()
    assert not hints._files
    assert hints._queue is None


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="macOS/BSD vnode events")
async def test_new_rollout_created_after_watch_is_discovered(tmp_path):
    hints = CompletionHints()
    hints.watch(set(), tmp_path)
    thread_id = str(uuid4())
    try:
        with (tmp_path / f"rollout-test-{thread_id}.jsonl").open("w") as writer:
            writer.write(record(str(uuid4())) + "\n")
            writer.flush()
            assert await asyncio.wait_for(hints.wait(), 1) == {thread_id}
    finally:
        hints.close()


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="macOS/BSD vnode events")
async def test_known_completion_does_not_wake_on_later_output(tmp_path):
    hints = CompletionHints()
    thread_id, turn_id = str(uuid4()), str(uuid4())
    path = tmp_path / f"rollout-test-{thread_id}.jsonl"
    path.write_text(record(turn_id) + "\n")
    hints.remember(thread_id, turn_id)
    hints.watch({path}, tmp_path)
    try:
        with path.open("a") as writer:
            writer.write(record(str(uuid4()), event="task_started") + "\n")
            writer.flush()
            waiter = asyncio.create_task(hints.wait())
            await asyncio.sleep(0.1)
            assert not waiter.done()
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    finally:
        hints.close()
