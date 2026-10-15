"""In-process fan-out hub for one companion publisher and desktop clients."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from itertools import count
from typing import Literal, cast
from uuid import UUID

from fish_memory.pet.protocol import PET_PROTOCOL_VERSION, unpack_pcm_frame
from fish_memory.pet.text_input import PetTextInput

_QueueKind = Literal["control", "pcm"]


@dataclass(frozen=True, slots=True)
class PetOutbound:
    kind: _QueueKind
    value: str | bytes


class PetEventHub:
    """Broadcast pet events without ever applying backpressure to audio."""

    def __init__(self, *, subscriber_queue_size: int = 128) -> None:
        if subscriber_queue_size < 2:
            raise ValueError("subscriber queue must contain at least two events")
        self._subscriber_queue_size = subscriber_queue_size
        self._subscribers: dict[int, asyncio.Queue[PetOutbound]] = {}
        self._subscriber_ids = count(1)
        self._control_only: set[int] = set()
        self._sequence = 0
        self._latest: dict[str, dict[str, object]] = {}
        self._publisher_count = 0
        self._closed = False
        self._interrupt_waiters: dict[str, asyncio.Future[None]] = {}
        self.text_input = PetTextInput()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def publisher_count(self) -> int:
        return self._publisher_count

    def publisher_connected(self) -> None:
        self._publisher_count += 1

    def publisher_disconnected(self) -> None:
        self._publisher_count = max(0, self._publisher_count - 1)

    def subscribe(
        self, *, control_only: bool = False
    ) -> tuple[int, asyncio.Queue[PetOutbound]]:
        if self._closed:
            raise RuntimeError("pet event hub is closed")
        subscriber_id = next(self._subscriber_ids)
        queue: asyncio.Queue[PetOutbound] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        self._subscribers[subscriber_id] = queue
        if control_only:
            self._control_only.add(subscriber_id)
        return subscriber_id, queue

    def reopen(self) -> None:
        """Allow an app object to enter a fresh lifespan in tests/reloads."""
        self._closed = False

    def unsubscribe(self, subscriber_id: int) -> None:
        self._subscribers.pop(subscriber_id, None)
        self._control_only.discard(subscriber_id)

    def snapshot(self) -> dict[str, object]:
        return {
            "version": PET_PROTOCOL_VERSION,
            "sequence": self._sequence,
            "type": "pet.snapshot",
            "generation": _latest_generation(self._latest),
            "timestampMs": round(time.time() * 1000),
            "payload": {
                "events": list(self._latest.values()),
                "publisherConnected": self._publisher_count > 0,
            },
        }

    def status(self) -> dict[str, object]:
        return {
            "version": PET_PROTOCOL_VERSION,
            "sequence": self._sequence,
            "publisherConnected": self._publisher_count > 0,
            "subscriberCount": len(self._subscribers),
            "latestEvents": sorted(self._latest),
        }

    def publish_control(
        self,
        event_type: str,
        generation: UUID | None,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("pet event hub is closed")
        if event_type == "pet.chat.result":
            self.text_input.complete(payload)
        self._sequence += 1
        event: dict[str, object] = {
            "version": PET_PROTOCOL_VERSION,
            "sequence": self._sequence,
            "type": event_type,
            "generation": str(generation) if generation is not None else None,
            "timestampMs": round(time.time() * 1000),
            "payload": dict(payload),
        }
        if event_type == "pet.playback.progress":
            key = event_type + ":" + str(payload["messageId"])
            previous = self._latest.get(key)
            if previous is not None:
                old = cast(dict[str, object], previous["payload"])
                if old["playbackId"] == payload["playbackId"] and cast(
                    int, old["revision"]
                ) >= cast(int, payload["revision"]):
                    return previous
            self._latest[key] = event
            keys = [
                key for key in self._latest if key.startswith("pet.playback.progress:")
            ]
            for key in keys[:-32]:
                del self._latest[key]
        if event_type in {
            "pet.state",
            "pet.companion.activity",
            "pet.plugin.sessions",
            "pet.microphone",
            "pet.microphone.state",
            "pet.speaker",
            "pet.speaker.state",
            "pet.expression",
            "pet.audio.start",
            "pet.audio.pause",
            "pet.audio.resume",
            "pet.audio.stop",
            "pet.generation.cancel",
            "pet.service.problems",
        }:
            self._latest[event_type] = event
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        for subscriber_id, queue in tuple(self._subscribers.items()):
            if not self._enqueue_control(queue, PetOutbound("control", encoded)):
                self.unsubscribe(subscriber_id)
        return event

    def publish_pcm(self, frame: bytes) -> bool:
        if self._closed:
            return False
        unpack_pcm_frame(frame)
        delivered = False
        outbound = PetOutbound("pcm", frame)
        for subscriber_id, queue in tuple(self._subscribers.items()):
            if subscriber_id in self._control_only:
                continue
            if queue.full():
                _discard_oldest_pcm(queue)
            if queue.full():
                continue
            queue.put_nowait(outbound)
            delivered = True
        return delivered

    async def request_answer_interrupt(self, fish_id: UUID, message_id: UUID) -> None:
        key = str(message_id)
        future = self._interrupt_waiters.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._interrupt_waiters[key] = future
        self.publish_control(
            "pet.answer.interrupt", None, {"fishId": str(fish_id), "messageId": key}
        )
        try:
            async with asyncio.timeout(2.0):
                await asyncio.shield(future)
        finally:
            if self._interrupt_waiters.get(key) is future:
                self._interrupt_waiters.pop(key, None)

    def terminal_answer_progress(
        self, fish_id: UUID, message_id: UUID
    ) -> dict[str, object] | None:
        """The Companion's frozen boundary, never an observer's guessed offset."""
        event = self._latest.get(f"pet.playback.progress:{message_id}")
        if event is None:
            return None
        progress = cast(dict[str, object], event["payload"])
        if progress.get("fishId") != str(fish_id) or progress.get("status") not in {
            "completed",
            "interrupted",
            "failed",
        }:
            return None
        return dict(progress)

    def answer_interrupted(self, message_id: UUID) -> None:
        future = self._interrupt_waiters.get(str(message_id))
        if future is not None and not future.done():
            future.set_result(None)

    async def close(self) -> None:
        self._closed = True
        self.text_input.close()
        self._subscribers.clear()
        self._control_only.clear()

    @staticmethod
    def _enqueue_control(
        queue: asyncio.Queue[PetOutbound],
        event: PetOutbound,
    ) -> bool:
        if queue.full():
            _discard_oldest_pcm(queue)
        if queue.full():
            # A client that cannot consume even ordered control messages must
            # reconnect and recover from the cached snapshot.
            return False
        queue.put_nowait(event)
        return True


def _discard_oldest_pcm(queue: asyncio.Queue[PetOutbound]) -> None:
    buffered: list[PetOutbound] = []
    discarded = False
    while not queue.empty():
        item = queue.get_nowait()
        if item.kind == "pcm" and not discarded:
            discarded = True
            continue
        buffered.append(item)
    for item in buffered:
        queue.put_nowait(item)


def _latest_generation(events: dict[str, dict[str, object]]) -> object:
    pet_events = [
        event
        for key, event in events.items()
        if not key.startswith("pet.playback.progress:")
        and key
        not in {
            "pet.companion.activity",
            "pet.plugin.sessions",
            "pet.microphone",
            "pet.microphone.state",
            "pet.speaker",
            "pet.speaker.state",
        }
    ]
    if not pet_events:
        return None
    latest = max(pet_events, key=_event_sequence)
    return latest.get("generation")


def _event_sequence(event: dict[str, object]) -> int:
    value = event.get("sequence")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1
