"""Bounded macOS rollout hints, without a polling thread or another process."""

from __future__ import annotations

import asyncio
import json
import os
import select
import sys
from collections import deque
from pathlib import Path
from typing import Any
from uuid import UUID


class CompletionHints:
    def __init__(self) -> None:
        self._queue: Any = None
        self._files: dict[Path, int] = {}
        self._paths: dict[int, Path] = {}
        self._directory: Path | None = None
        self._dirty: set[Path] = set()
        self._changed = asyncio.Event()
        self._seen: deque[tuple[str, str]] = deque(maxlen=1024)

    def watch(self, files: set[Path], directory: Path) -> None:
        if sys.platform != "darwin":
            return  # Other hosts keep the existing low-frequency API scan.
        if self._queue is None:
            self._queue = select.kqueue()
            asyncio.get_running_loop().add_reader(self._queue.fileno(), self._ready)
        if self._directory != directory:
            if self._directory is not None:
                self._remove(self._directory)
            if self._add(directory):
                self._directory = directory
                self._discover(mark_dirty=False)
        for path in files:
            self._add(path)

    def remember(self, thread_id: str, turn_id: str) -> None:
        hint = (thread_id, turn_id)
        if hint not in self._seen:
            self._seen.append(hint)

    def _add(self, path: Path) -> bool:
        if sys.platform != "darwin":
            return False  # Only watch() creates the queue, on macOS.
        if path in self._files:
            return True
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        try:
            self._queue.control(
                [
                    select.kevent(
                        fd,
                        filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        fflags=select.KQ_NOTE_WRITE
                        | select.KQ_NOTE_EXTEND
                        | select.KQ_NOTE_DELETE
                        | select.KQ_NOTE_RENAME,
                    )
                ],
                0,
                0,
            )
        except OSError:
            os.close(fd)
            return False
        self._files[path] = fd
        self._paths[fd] = path
        if len(self._files) > 128:
            oldest = next(p for p in self._files if p != self._directory and p != path)
            self._remove(oldest)
        return True

    def _remove(self, path: Path) -> None:
        fd = self._files.pop(path, None)
        if fd is not None:
            self._paths.pop(fd, None)
            os.close(fd)  # Closing unregisters this vnode from kqueue.

    def _discover(self, *, mark_dirty: bool) -> None:
        if self._directory is None:
            return
        try:
            paths = sorted(self._directory.glob("rollout-*.jsonl"))[-127:]
            for path in paths:
                if path not in self._files and self._add(path) and mark_dirty:
                    self._dirty.add(path)
        except OSError:
            pass

    def _ready(self) -> None:
        # Non-blocking drain only when the kernel signals readiness. There is no
        # periodic filesystem scan. Unlike FSEvents, vnode writes do not wait
        # for Codex to close its long-lived rollout writer.
        if sys.platform != "darwin":
            return
        for event in self._queue.control(None, 128, 0):
            path = self._paths.get(event.ident)
            if path is None:
                continue
            if path == self._directory:
                self._discover(mark_dirty=True)
            else:
                self._dirty.add(path)
            if event.fflags & (select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME):
                self._remove(path)
                if path == self._directory:
                    self._directory = None
        if self._dirty:
            self._changed.set()

    async def wait(self) -> set[str]:
        while True:
            await self._changed.wait()
            await asyncio.sleep(0.05)  # Coalesce a burst, never poll idle files.
            self._changed.clear()
            paths, self._dirty = self._dirty, set()
            hints = await asyncio.to_thread(rollout_completion_hints, paths)
            fresh = {hint for hint in hints if hint not in self._seen}
            self._seen.extend(fresh)
            if fresh:
                return {thread_id for thread_id, _ in fresh}

    def close(self) -> None:
        if self._queue is not None:
            asyncio.get_running_loop().remove_reader(self._queue.fileno())
            self._queue.close()
            self._queue = None
        for path in list(self._files):
            self._remove(path)
        self._directory = None
        self._dirty.clear()
        self._changed.clear()


def rollout_signature(path: str | None) -> tuple[int, int, int, int] | None:
    if not path:
        return None
    try:
        value = Path(path).stat()
        return value.st_dev, value.st_ino, value.st_mtime_ns, value.st_size
    except OSError:
        return None


def rollout_completion_hints(paths: set[Path]) -> set[tuple[str, str]]:
    """Read bounded changed tails as hints; API completion stays authoritative."""
    found: set[tuple[str, str]] = set()
    for file in paths:
        if not file.name.startswith("rollout-") or file.suffix != ".jsonl":
            continue
        try:
            thread_id = str(UUID(file.stem[-36:]))
            with file.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                start = max(0, size - 64 * 1024)
                handle.seek(start)
                lines = handle.read(64 * 1024).split(b"\n")
            for line in lines[1 if start else 0 : -1]:
                if b'"task_complete"' not in line:
                    continue
                try:
                    record = json.loads(line)
                    payload = record.get("payload", {})
                    if (
                        record.get("type") == "event_msg"
                        and payload.get("type") == "task_complete"
                    ):
                        found.add((thread_id, str(UUID(payload["turn_id"]))))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
        except (OSError, ValueError):
            continue
    return found
