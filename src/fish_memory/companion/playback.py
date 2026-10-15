"""Exclusive live-playback ownership for the native Companion."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

PlaybackKind = Literal["answer", "home", "agent", "proactive", "wake"]


class PlaybackArbiterClosed(RuntimeError):
    """Raised when live playback is requested after shutdown."""


@dataclass(frozen=True, slots=True)
class PlaybackOwner:
    token: UUID
    kind: PlaybackKind
    message_id: UUID | None


@dataclass(frozen=True, slots=True)
class ForegroundReservation:
    token: UUID


@dataclass(slots=True)
class _PlaybackWaiter:
    kind: PlaybackKind
    message_id: UUID | None
    future: asyncio.Future[PlaybackOwner]
    granted_owner: PlaybackOwner | None = None


class PlaybackArbiter:
    """Serialize all live PCM users without owning their business state."""

    def __init__(
        self,
        *,
        token_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._token_factory = token_factory
        self._current: PlaybackOwner | None = None
        self._current_tts: BoundTtsOutput | None = None
        self._waiters: deque[_PlaybackWaiter] = deque()
        self._foreground: set[ForegroundReservation] = set()
        self._spoken_text = ""
        self._closed = False
        self._operation_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._interrupt_lock = asyncio.Lock()
        self._pcm_interrupt_lock = asyncio.Lock()

    @property
    def current_owner(self) -> PlaybackOwner | None:
        return self._current

    @property
    def current_tts(self) -> BoundTtsOutput | None:
        return self._current_tts

    @property
    def current_spoken_text(self) -> str:
        return self._spoken_text

    @property
    def foreground_reserved(self) -> bool:
        return bool(self._foreground)

    async def acquire(
        self,
        kind: PlaybackKind,
        message_id: UUID | None,
    ) -> PlaybackOwner:
        if kind not in {"answer", "home", "agent", "proactive", "wake"}:
            raise ValueError("unsupported playback owner kind")
        if self._closed:
            raise PlaybackArbiterClosed("playback arbiter is closed")
        future = asyncio.get_running_loop().create_future()
        waiter = _PlaybackWaiter(kind, message_id, future)
        self._waiters.append(waiter)
        self._grant_next()
        try:
            return await future
        except asyncio.CancelledError:
            with suppress(ValueError):
                self._waiters.remove(waiter)
            if waiter.granted_owner is not None:
                await self.release(waiter.granted_owner)
            else:
                self._grant_next()
            raise

    def try_acquire(
        self, kind: PlaybackKind, message_id: UUID | None
    ) -> PlaybackOwner | None:
        """Admit a live opportunity only while idle, without creating a waiter."""
        if kind not in {"answer", "home", "agent", "proactive", "wake"}:
            raise ValueError("unsupported playback owner kind")
        if self._closed:
            raise PlaybackArbiterClosed("playback arbiter is closed")
        if self._current is not None or self._waiters or self._foreground:
            return None
        self._current = PlaybackOwner(self._token_factory(), kind, message_id)
        self._spoken_text = ""
        return self._current

    async def release(self, owner: PlaybackOwner) -> bool:
        async with self._control_lock:
            async with self._interrupt_lock:
                async with self._pcm_interrupt_lock:
                    async with self._operation_lock:
                        if not self.is_current(owner):
                            return False
                        self._current = None
                        self._current_tts = None
                        self._spoken_text = ""
                        self._grant_next()
                        return True

    def is_current(self, owner: PlaybackOwner) -> bool:
        return not self._closed and self._current == owner

    def reserve_foreground(self) -> ForegroundReservation:
        if self._closed:
            raise PlaybackArbiterClosed("playback arbiter is closed")
        reservation = ForegroundReservation(self._token_factory())
        self._foreground.add(reservation)
        return reservation

    def release_foreground(self, reservation: ForegroundReservation) -> bool:
        if reservation not in self._foreground:
            return False
        self._foreground.remove(reservation)
        self._grant_next()
        return True

    def set_spoken_text(self, owner: PlaybackOwner, text: str) -> bool:
        if not self.is_current(owner):
            return False
        self._spoken_text = text
        return True

    def bind_tts(self, owner: PlaybackOwner, output: Any) -> BoundTtsOutput:
        if isinstance(output, BoundTtsOutput):
            if output.matches(self, owner):
                if self.is_current(owner):
                    self._current_tts = output
                return output
            raise RuntimeError("TTS output is bound to another playback owner")
        bound = BoundTtsOutput(self, owner, output)
        if self.is_current(owner):
            self._current_tts = bound
        return bound

    def bind_pcm(self, owner: PlaybackOwner, sink: Any) -> BoundPcmSink:
        return BoundPcmSink(self, owner, sink)

    async def run_if_current(
        self,
        owner: PlaybackOwner,
        operation: Callable[[], object],
    ) -> tuple[bool, object | None]:
        """Run one live-resource operation while release is serialized."""
        async with self._operation_lock:
            if not self.is_current(owner):
                return False, None
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            return True, result

    async def run_control_if_current(
        self,
        owner: PlaybackOwner,
        operation: Callable[[], object],
    ) -> tuple[bool, object | None]:
        """Serialize provider control operations against release and close."""
        async with self._control_lock:
            if not self.is_current(owner):
                return False, None
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            return True, result

    async def run_interrupt_if_current(
        self,
        owner: PlaybackOwner,
        operation: Callable[[], object],
    ) -> tuple[bool, object | None]:
        """Run owner revocation while normal control work unwinds."""
        async with self._interrupt_lock:
            if not self.is_current(owner):
                return False, None
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            return True, result

    async def run_pcm_interrupt_if_current(
        self,
        owner: PlaybackOwner,
        operation: Callable[[], object],
    ) -> tuple[bool, object | None]:
        """Let PCM controls bypass a backpressured append without nesting TTS locks."""
        async with self._pcm_interrupt_lock:
            if not self.is_current(owner):
                return False, None
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            return True, result

    async def close(self) -> None:
        async with self._control_lock:
            async with self._interrupt_lock:
                async with self._pcm_interrupt_lock:
                    async with self._operation_lock:
                        if self._closed:
                            return
                        self._closed = True
                        self._current = None
                        self._current_tts = None
                        self._spoken_text = ""
                        self._foreground.clear()
                        error = PlaybackArbiterClosed("playback arbiter is closed")
                        while self._waiters:
                            waiter = self._waiters.popleft()
                            if not waiter.future.done():
                                waiter.future.set_exception(error)

    def _grant_next(self) -> None:
        if self._closed or self._current is not None or self._foreground:
            return
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.future.done():
                continue
            owner = PlaybackOwner(
                self._token_factory(),
                waiter.kind,
                waiter.message_id,
            )
            self._current = owner
            self._current_tts = None
            self._spoken_text = ""
            waiter.granted_owner = owner
            waiter.future.set_result(owner)
            return


class BoundTtsOutput:
    """Owner-bound facade for every operation that can affect live speech."""

    def __init__(
        self,
        arbiter: PlaybackArbiter,
        owner: PlaybackOwner,
        output: Any,
    ) -> None:
        self._arbiter = arbiter
        self._owner = owner
        self._output = output
        bind_playback = getattr(output, "bind_playback", None)
        if callable(bind_playback):
            bind_playback(arbiter, owner)

    @property
    def owner(self) -> PlaybackOwner:
        return self._owner

    def matches(self, arbiter: PlaybackArbiter, owner: PlaybackOwner) -> bool:
        return self._arbiter is arbiter and self._owner == owner

    @property
    def played_offset(self) -> int:
        if not self._arbiter.is_current(self._owner):
            return 0
        value = getattr(self._output, "played_offset", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def display_offset(self) -> int:
        if not self._arbiter.is_current(self._owner):
            return 0
        return getattr(self._output, "display_offset", self.played_offset)

    @property
    def consumption_is_confirmed(self) -> bool:
        return getattr(self._output, "consumption_is_confirmed", False)

    def pop_facial_cue(self) -> dict[str, object] | None:
        pop = getattr(self._output, "pop_facial_cue", None)
        if self._arbiter.is_current(self._owner) and callable(pop):
            return pop()
        return None

    @property
    def supports_interrupt(self) -> bool:
        return callable(getattr(self._output, "interrupt", None))

    async def start(self, *, text_offset: int = 0) -> None:
        async def operation() -> None:
            if text_offset:
                await self._output.start(text_offset=text_offset)
            else:
                await self._output.start()

        await self._arbiter.run_control_if_current(self._owner, operation)

    async def append(self, text: str) -> None:
        await self._run("append", text)

    def set_speech_tags(self, enabled: bool) -> None:
        configure = getattr(self._output, "set_speech_tags", None)
        if self._arbiter.is_current(self._owner) and callable(configure):
            configure(enabled)

    async def flush(self) -> None:
        await self._run("flush")

    def activate_diagnostics(self, diagnostics: Any) -> None:
        activate = getattr(self._output, "activate_diagnostics", None)
        if self._arbiter.is_current(self._owner) and callable(activate):
            activate(diagnostics)

    async def finish(self) -> None:
        await self._run("finish")

    async def stop(self) -> None:
        await self._run_interrupt("stop")

    async def interrupt(self) -> None:
        interrupt = getattr(self._output, "interrupt", None)
        if not callable(interrupt):
            interrupt = self._output.stop
        await self._arbiter.run_interrupt_if_current(self._owner, interrupt)

    async def pause(self) -> None:
        await self._run_interrupt("pause")

    async def resume(self) -> None:
        await self._run_interrupt("resume")

    async def duck(self) -> None:
        await self._run_interrupt("duck")

    async def unduck(self) -> None:
        await self._run_interrupt("unduck")

    async def _run(self, name: str, *args: object) -> None:
        operation = getattr(self._output, name)
        await self._arbiter.run_control_if_current(
            self._owner,
            lambda: operation(*args),
        )

    async def _run_interrupt(self, name: str, *args: object) -> None:
        operation = getattr(self._output, name)
        await self._arbiter.run_interrupt_if_current(
            self._owner,
            lambda: operation(*args),
        )


class BoundPcmSink:
    """Owner-bound facade used by provider receive loops and draft promotion."""

    def __init__(
        self,
        arbiter: PlaybackArbiter,
        owner: PlaybackOwner,
        sink: Any,
    ) -> None:
        self._arbiter = arbiter
        self._owner = owner
        self._sink = sink

    @property
    def owner(self) -> PlaybackOwner:
        return self._owner

    @property
    def played_offset(self) -> int:
        if not self._arbiter.is_current(self._owner):
            return 0
        value = getattr(self._sink, "played_offset", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def played_audio_bytes(self) -> int:
        if not self._arbiter.is_current(self._owner):
            return 0
        value = getattr(self._sink, "played_audio_bytes", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def display_offset(self) -> int:
        if not self._arbiter.is_current(self._owner):
            return 0
        return getattr(self._sink, "display_offset", self.played_offset)

    async def start(self) -> None:
        await self._run("start")

    def reset(self, text_offset: int = 0) -> None:
        if not self._arbiter.is_current(self._owner):
            return
        reset = getattr(self._sink, "reset", None)
        if callable(reset):
            try:
                reset(text_offset)
            except TypeError:
                if text_offset:
                    raise
                reset()

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        append_owned = getattr(self._sink, "append_owned", None)
        if callable(append_owned):

            async def operation() -> None:
                await append_owned(
                    self._owner,
                    pcm,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
        else:

            async def operation() -> None:
                await self._sink.append(
                    pcm,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )

        await self._arbiter.run_if_current(self._owner, operation)

    async def stop(self) -> None:
        await self._run_interrupt("stop")

    async def pause(self) -> None:
        await self._run_interrupt("pause")

    async def resume(self) -> None:
        await self._run_interrupt("resume")

    async def duck(self) -> None:
        await self._run_interrupt("duck")

    async def unduck(self) -> None:
        await self._run_interrupt("unduck")

    async def mark(self, end_offset: int) -> None:
        await self._arbiter.run_if_current(
            self._owner, lambda: self._sink.mark(end_offset)
        )

    async def wait_until_drained(self) -> None:
        # Draining is a passive, potentially turn-long wait. Holding the
        # arbiter's mutation lock here prevents pause/duck/interrupt from
        # reaching the physical sink until playback has already completed.
        # Snapshot the immutable owner-bound operation, then wait lock-free.
        if not self._arbiter.is_current(self._owner):
            return
        wait_for_owner = getattr(self._sink, "wait_until_owner_drained", None)
        if callable(wait_for_owner):
            await wait_for_owner(self._owner)
            return
        await self._sink.wait_until_drained()

    async def _run(self, name: str) -> None:
        operation = getattr(self._sink, name)
        await self._arbiter.run_if_current(self._owner, operation)

    async def _run_interrupt(self, name: str) -> None:
        operation = getattr(self._sink, name)
        await self._arbiter.run_pcm_interrupt_if_current(self._owner, operation)
