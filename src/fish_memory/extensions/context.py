"""Read published pet material; never export the model's system messages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fish_memory.answer.context import DEFAULT_RECENT_CONTEXT_LIMIT
from fish_memory.answer.snapshot import render_static_profile
from fish_memory.clock import format_prompt_timestamp
from fish_memory.extensions.event_context import event_history_view
from fish_memory.models import Message, visible_message_content


class ExtensionContext:
    DEFAULT_FIELDS = frozenset(
        {"personality", "memory", "history", "perception", "events"}
    )
    FIELDS = DEFAULT_FIELDS | {"audiovisual"}

    def __init__(
        self,
        fishes: Any,
        history: Any,
        snapshots: Any,
        environment: Any,
        events: Any,
        *,
        history_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
        snapshot_max_age: timedelta = timedelta(days=7),
        audiovisual: Any = None,
        session_history: Any = None,
    ):
        self.fishes, self.history, self.snapshots = fishes, history, snapshots
        self.environment, self.events = environment, events
        self.audiovisual = audiovisual
        self.session_history = session_history
        self.history_limit = min(history_limit, DEFAULT_RECENT_CONTEXT_LIMIT)
        self.snapshot_max_age = snapshot_max_age
        self._cache: dict[
            tuple[UUID, UUID], tuple[float, tuple[dict[str, Any], list[Message]]]
        ] = {}
        self._loading: dict[
            tuple[UUID, UUID], asyncio.Task[tuple[dict[str, Any], list[Message]]]
        ] = {}

    async def read(
        self, fish: UUID, session: UUID, fields: list[str] | None = None
    ) -> dict[str, Any]:
        if fields is not None and (
            not isinstance(fields, list) or not all(isinstance(f, str) for f in fields)
        ):
            raise ValueError("fields must be a list of context field names")
        requested = set(fields) if fields is not None else self.DEFAULT_FIELDS
        if not requested <= self.FIELDS:
            raise ValueError("Unknown context field")
        if not requested:
            return {}
        # Auxiliary reasoning reads only what it requested. In particular a
        # topic-only request must not load personal, image, or activity context.
        value, _history = await self._load_selected(
            fish, session, requested, event_limit=16
        )
        return value

    async def read_for_speech(
        self, fish: UUID, session: UUID
    ) -> tuple[dict[str, Any], list[Message]]:
        """Keep native messages private; speech uses the ordinary chat builder."""
        key = (fish, session)
        cached = self._cache.get(key)
        # One second bounds DB load; perception supplies its own capture timestamps.
        if cached is None or time.monotonic() - cached[0] > 1:
            task = self._loading.get(key)
            if task is None:
                task = asyncio.create_task(self._load(fish, session))
                self._loading[key] = task
            try:
                value = await asyncio.shield(task)
            finally:
                if task.done():
                    self._loading.pop(key, None)
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (time.monotonic(), value)
        else:
            value = cached[1]
        return value

    def invalidate(self, fish: UUID, session: UUID) -> None:
        self._cache.pop((fish, session), None)

    async def _load(
        self, fish: UUID, session: UUID
    ) -> tuple[dict[str, Any], list[Message]]:
        return await self._load_selected(fish, session, self.DEFAULT_FIELDS)

    async def _load_selected(
        self,
        fish: UUID,
        session: UUID,
        requested: set[str] | frozenset[str],
        *,
        event_limit: int = 12,
    ) -> tuple[dict[str, Any], list[Message]]:
        reads = {}
        if "personality" in requested:
            reads["soul"] = self.fishes.get_profile(fish)
        if "history" in requested:
            reads["history"] = self.history.load_history_readonly(
                fish, session, before_sequence=(1 << 63) - 1, limit=self.history_limit
            )
            if self.session_history is not None:
                reads["history_text"] = self.session_history.load(fish, session)
        if "memory" in requested:
            reads["static"] = self.snapshots.load_normalized_static_profile(fish)
        if "perception" in requested:
            reads["perception"] = self.environment.load_cached(
                fish,
                include_image=False,
                **({"include_summary": False} if self.session_history else {}),
            )
        if "events" in requested:
            reads["events"] = self.events.recent(fish, session, event_limit)
        if "audiovisual" in requested and self.audiovisual is not None:
            reads["audiovisual"] = self.audiovisual.get_latest_media_context_summary(
                fish
            )
        loaded = dict(zip(reads, await asyncio.gather(*reads.values()), strict=True))
        history = loaded.get("history", [])
        result: dict[str, Any] = {}
        if "personality" in requested:
            soul = loaded["soul"]
            result["personality"] = asdict(soul) if soul is not None else None
        if "history" in requested:
            if "history_text" in loaded:
                result["history_text"] = loaded["history_text"]
            result["history"] = [
                {
                    "id": str(m.id),
                    "role": m.role.value,
                    "content": visible_message_content(m),
                    "created_at": format_prompt_timestamp(m.created_at),
                }
                for m in history
                if visible_message_content(m)
            ]
        if "memory" in requested:
            static = loaded["static"]
            result["memory"] = {
                "static": render_static_profile(static) if static else "",
            }
        if "perception" in requested:
            perception = loaded["perception"]
            result["perception"] = {
                "available": perception.enabled,
                "text": perception.summary_context
                if self.session_history is None
                else "",
            }
        if "audiovisual" in requested:
            media = loaded.get("audiovisual")
            result["audiovisual"] = {
                "available": media is not None,
                "text": media.summary if media else "",
            }
        if "events" in requested:
            result["events"] = [
                {k: event.get(k) for k in ("type", "data", "status", "received_at")}
                for event in event_history_view(loaded["events"], {})
                if event.get("include_in_context", True)
            ]
        encoded = json.dumps(result, sort_keys=True, default=str, ensure_ascii=False)
        result["revision"] = hashlib.sha256(encoded.encode()).hexdigest()
        result["as_of"] = format_prompt_timestamp(datetime.now(UTC))
        return result, history

    async def close(self) -> None:
        for task in self._loading.values():
            task.cancel()
        await asyncio.gather(*self._loading.values(), return_exceptions=True)
        self._loading.clear()
        self._cache.clear()
