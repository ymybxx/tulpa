"""Incremental completion discovery and durable at-most-once message identity."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fish_memory.codex_notifications.changes import CompletionHints, rollout_signature
from fish_memory.codex_notifications.client import CodexReadClient

logger = logging.getLogger(__name__)


def completion(
    thread: dict[str, Any], turn: dict[str, Any], since: float
) -> dict[str, Any] | None:
    """Never infer completion from idle/notLoaded or partial assistant output."""
    if turn.get("status") != "completed" or turn.get("error"):
        return None
    completed_at = turn.get("completedAt")
    if not isinstance(completed_at, (float, int)) or completed_at < since:
        return None
    if thread.get("parentThreadId") or str(thread.get("source", "")).startswith(
        "subAgent"
    ):
        return None
    thread_id, turn_id = thread.get("id"), turn.get("id")
    if not all(isinstance(v, str) and 0 < len(v) < 200 for v in (thread_id, turn_id)):
        return None
    texts = [
        i["text"]
        for i in turn.get("items", [])
        if i.get("type") == "agentMessage"
        and i.get("phase") == "final_answer"
        and isinstance(i.get("text"), str)
    ]
    text = "\n".join(texts).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("disposition") == "input_required":
            return None
    except ValueError:
        pass
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "title": str(thread.get("name") or thread.get("preview") or "Codex 任务")[:200],
        "completed_at": completed_at,
        "result": text[:8000],
    }


class CompletionObserver:
    def __init__(
        self,
        client: CodexReadClient,
        state_path: Path,
        deliver: Callable[[dict[str, Any], str], Awaitable[bool]],
        *,
        enabled: bool = False,
        now: Callable[[], float] = time.time,
        interval: float = 5,
    ) -> None:
        self.client, self.path, self.deliver = client, state_path, deliver
        self.enabled, self.now, self.interval = enabled, now, interval
        self.state: dict[str, Any] | None = None
        self.error: str | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # A listening session never catches up on completions from downtime.
        # Capture its boundary before asynchronous startup/polling can be delayed.
        self._since = self.now()
        self._reset = enabled
        self._generation = 0
        self._turn_versions: dict[str, tuple[Any, ...]] = {}
        self._hints = CompletionHints()
        self._home = str(
            Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()
        )

    def start(self) -> None:
        if self._task is not None:
            return
        self._since = self.now()
        self._reset = self.enabled
        self._task = asyncio.create_task(self._run(), name="codex-completion-observer")

    def configure(self, enabled: bool) -> None:
        if self.enabled != enabled:
            self.enabled = enabled
            self._generation += 1
            self._reset = True
            if enabled:
                self._since = self.now()
            self._wake.set()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "error": self.error,
            "pending": len(self.state.get("pending", {})) if self.state else 0,
        }

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("version") != 1 or not isinstance(value.get("pending"), dict):
                raise ValueError("invalid notification state")
            return value
        return {
            "version": 1,
            "since": self.now(),
            "cursor": self.now(),
            "seen": [],
            "pending": {},
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(self.state, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    async def poll(self, thread_ids: set[str] | None = None) -> None:
        if not self.enabled:
            return
        generation = self._generation
        baseline_changed = self.state is None
        if self.state is None:
            self.state = await asyncio.to_thread(self._load)
        if self._reset or self.state.get("paused"):
            self.state.update(since=self._since, cursor=self._since, pending={})
            self.state.pop("paused", None)
            self._reset = False
            baseline_changed = True
            self._turn_versions.clear()
        # Keep seen IDs for dedupe, but reset both the scan cursor and undelivered
        # backlog on every startup/enable. An API reconnect within this session
        # keeps its cursor so completions during a transient outage are not lost.
        if baseline_changed:
            await asyncio.to_thread(self._save)
        started = self.now()
        cutoff = max(self.state["since"], self.state["cursor"] - 60)
        cursor: str | None = None
        sessions = Path(self._home) / "sessions"
        files: set[Path] = set()
        # Fully page the changed interval. Do not advance the watermark on an error.
        while True:
            result: dict[str, Any]
            if thread_ids is None:
                result = await self.client.read(
                    "thread/list",
                    {
                        "sortKey": "updated_at",
                        "sortDirection": "desc",
                        "limit": 100,
                        "sourceKinds": [
                            "cli",
                            "vscode",
                            "exec",
                            "appServer",
                            "unknown",
                        ],
                        "useStateDbOnly": True,
                        **({"cursor": cursor} if cursor else {}),
                    },
                )
            else:
                hinted = []
                for thread_id in thread_ids:
                    try:
                        value = await self.client.read(
                            "thread/read", {"threadId": thread_id}
                        )
                        hinted.append(value["thread"])
                    except RuntimeError:
                        continue  # A local file hint need not be a known task.
                result = {"data": hinted}
            reached = False
            for thread in result.get("data", []):
                if thread_ids is None and thread.get("updatedAt", 0) < cutoff:
                    reached = True
                    break
                if not self.enabled or generation != self._generation:
                    return
                path = thread.get("path")
                if isinstance(path, str) and Path(path).is_relative_to(sessions):
                    files.add(Path(path))
                signature = await asyncio.to_thread(rollout_signature, path)
                version = (thread.get("updatedAt"), signature)
                if (
                    thread_ids is None
                    and signature
                    and self._turn_versions.get(thread["id"]) == version
                ):
                    continue
                turns = await self.client.read(
                    "thread/turns/list",
                    {
                        "threadId": thread["id"],
                        "limit": 100,
                        "itemsView": "summary",
                        "sortDirection": "desc",
                    },
                )
                turn_cursor = None
                settled = True
                while True:
                    old = False
                    for turn in turns.get("data", []):
                        if turn.get("status") == "completed":
                            self._hints.remember(thread["id"], turn["id"])
                        settled &= turn.get("status") in {
                            "completed",
                            "interrupted",
                            "failed",
                        }
                        event = completion(thread, turn, cutoff)
                        if event:
                            key = str(
                                uuid5(
                                    NAMESPACE_URL,
                                    f"fish:codex:{self._home}:{event['thread_id']}:{event['turn_id']}",
                                )
                            )
                            if key not in self.state["seen"]:
                                if key not in self.state["pending"]:
                                    self.state["pending"][key] = event
                                    logger.info(
                                        "Codex completion detected "
                                        "message=%s lag_ms=%.0f",
                                        key,
                                        (self.now() - event["completed_at"]) * 1000,
                                    )
                        ended = turn.get("completedAt")
                        if isinstance(ended, (int, float)) and ended < cutoff:
                            old = True
                    turn_cursor = turns.get("nextCursor")
                    if not turn_cursor or old:
                        break
                    turns = await self.client.read(
                        "thread/turns/list",
                        {
                            "threadId": thread["id"],
                            "limit": 100,
                            "itemsView": "summary",
                            "sortDirection": "desc",
                            "cursor": turn_cursor,
                        },
                    )
                if signature and settled:
                    self._turn_versions[thread["id"]] = version
                    if len(self._turn_versions) > 1024:
                        del self._turn_versions[next(iter(self._turn_versions))]
                else:
                    self._turn_versions.pop(thread["id"], None)
            cursor = result.get("nextCursor")
            if reached or not cursor:
                break
        if thread_ids is None:
            self.state["cursor"] = started
        await asyncio.to_thread(self._save)
        for key, event in list(self.state["pending"].items()):
            if not self.enabled or generation != self._generation:
                return
            if not await self.deliver(event, key):
                break
            del self.state["pending"][key]
            self.state["seen"] = (self.state["seen"] + [key])[-10000:]
            await asyncio.to_thread(self._save)
            # The common chat/playback queue owns pacing, not completion discovery.
        self._hints.watch(files, sessions / datetime.now().strftime("%Y/%m/%d"))

    async def _run(self) -> None:
        thread_ids: set[str] | None = None
        next_full_poll = 0.0
        try:
            while True:
                self._wake.clear()
                if not self.enabled:
                    self._hints.close()
                    await self.client.close()
                    if self._reset:
                        if self.state is None:
                            self.state = await asyncio.to_thread(self._load)
                        self.state.update(paused=True, pending={})
                        await asyncio.to_thread(self._save)
                        self._reset = False
                    await self._wake.wait()
                    continue
                try:
                    if time.monotonic() >= next_full_poll or self._reset:
                        thread_ids = None
                    await self.poll(thread_ids)
                    if thread_ids is None:
                        next_full_poll = time.monotonic() + self.interval
                    self.error = None
                except Exception as error:
                    self.error = str(error)[:200]
                    logger.warning(
                        "Codex completion observer: %s", type(error).__name__
                    )
                    await self.client.close()
                try:
                    wake = asyncio.create_task(self._wake.wait())
                    completion_hint = asyncio.create_task(self._hints.wait())
                    await asyncio.wait(
                        (wake, completion_hint),
                        timeout=30
                        if self.error
                        else max(0, next_full_poll - time.monotonic()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    thread_ids = (
                        completion_hint.result() if completion_hint.done() else None
                    )
                finally:
                    wake.cancel()
                    completion_hint.cancel()
                    await asyncio.gather(wake, completion_hint, return_exceptions=True)
        finally:
            self._hints.close()
            await self.client.close()

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        else:
            self._hints.close()
            await self.client.close()
