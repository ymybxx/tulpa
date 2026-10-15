"""Private PCM buffering and the one-way handoff to an owned playback sink."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from threading import Lock

from fish_memory.companion.playback import (
    BoundPcmSink,
    PlaybackArbiter,
    PlaybackOwner,
)
from fish_memory.companion.tts import PcmSink

logger = logging.getLogger(__name__)


class PromotablePcmSink:
    """Keep one exact text-prefix PCM stream private until public promotion."""

    def __init__(
        self,
        live_sink: PcmSink,
        max_buffer_bytes: int,
        *,
        on_event: Callable[[str, dict[str, object]], None] | None = None,
        playback_arbiter: PlaybackArbiter | None = None,
    ) -> None:
        if max_buffer_bytes < 0:
            raise ValueError("max_buffer_bytes must not be negative")
        self._live_sink = live_sink
        self._playback_arbiter = playback_arbiter
        self._playback_owner: PlaybackOwner | None = None
        self._bound_live_sink: PcmSink | BoundPcmSink = live_sink
        self._on_event = on_event
        self._max_buffer_bytes = max_buffer_bytes
        self._lock = Lock()
        self._io_gate = asyncio.Lock()
        self._state = "buffering"
        self._packets: list[tuple[bytes, int, int]] = []
        self._buffered_bytes = 0
        self._first_start_offset: int | None = None
        self._last_range: tuple[int, int] | None = None
        self._live_attached = False

    @property
    def played_offset(self) -> int:
        with self._lock:
            promoted = self._state == "promoted"
        return self._bound_live_sink.played_offset if promoted else 0

    @property
    def played_audio_bytes(self) -> int:
        with self._lock:
            promoted = self._state == "promoted"
        value = (
            getattr(self._bound_live_sink, "played_audio_bytes", 0) if promoted else 0
        )
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def display_offset(self) -> int:
        with self._lock:
            promoted = self._live_attached and self._state in {"attaching", "promoted"}
        return self._bound_live_sink.display_offset if promoted else 0

    async def mark(self, end_offset: int) -> None:
        async with self._io_gate:
            with self._lock:
                if self._state == "buffering":
                    if self._last_range is None or end_offset > self._last_range[1]:
                        return
                    self._packets.append((b"", end_offset, end_offset))
                    return
                if self._state != "promoted":
                    return
            await self._bound_live_sink.mark(end_offset)

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return self._buffered_bytes

    @property
    def retirement_may_mutate_live_sink(self) -> bool:
        with self._lock:
            return self._live_attached or self._state in {"attaching", "revoking"}

    def bind_playback_sink(
        self,
        arbiter: PlaybackArbiter,
        owner: PlaybackOwner,
    ) -> "PromotablePcmSink":
        """Reuse the promotion's existing owner layer for provider callbacks."""
        with self._lock:
            if (
                self._playback_arbiter is not arbiter
                or self._playback_owner != owner
                or self._state not in {"attaching", "promoted"}
            ):
                raise RuntimeError("promotable PCM sink has no matching live owner")
        return self

    async def start(self) -> None:
        with self._lock:
            if self._state == "promoted":
                promoted = True
            elif self._state == "buffering":
                promoted = False
                self._clear_locked()
            else:
                return
        if promoted:
            await self._bound_live_sink.start()

    def reset(self) -> None:
        with self._lock:
            state = self._state
            if state == "buffering":
                self._clear_locked()
                return
            if state in {"attaching", "promoted"}:
                self._state = "revoking"
                self._clear_locked()
                reset_live = self._live_attached
            else:
                reset_live = False
        if reset_live:
            self._reset_live()

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        if not pcm:
            return
        with self._lock:
            state = self._state
            if state == "buffering":
                if not self._accept_range_locked(
                    pcm,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    enforce_cap=True,
                ):
                    self._discard_locked(reason="range_mismatch")
                    return
                first_packet = self._buffered_bytes == 0
                self._packets.append((bytes(pcm), start_offset, end_offset))
                self._buffered_bytes += len(pcm)
                if first_packet:
                    self._emit(
                        "voice_draft_tts_first_audio_buffered",
                        {"pcmBytes": len(pcm)},
                    )
                return
            if state not in {"attaching", "promoted"}:
                return

        async with self._io_gate:
            with self._lock:
                if self._state not in {"attaching", "promoted"}:
                    return
                if not self._accept_range_locked(
                    pcm,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    enforce_cap=False,
                ):
                    self._discard_locked(reason="range_mismatch")
                    reset_live = self._live_attached
                else:
                    reset_live = False
            if reset_live:
                self._reset_live()
                return
            await self._bound_live_sink.append(
                pcm,
                start_offset=start_offset,
                end_offset=end_offset,
            )

    async def promote(
        self,
        *,
        expected_end_offset: int,
        owner: PlaybackOwner | None = None,
        before_playback: Callable[[], None] | None = None,
    ) -> bool:
        async with self._io_gate:
            arbiter = self._playback_arbiter
            if arbiter is not None:
                if owner is None or not arbiter.is_current(owner):
                    return False
                self._playback_owner = owner
                self._bound_live_sink = arbiter.bind_pcm(owner, self._live_sink)
            with self._lock:
                if self._state in {"attaching", "promoted"}:
                    return False
                has_pcm = bool(self._packets)
                if (
                    self._state != "buffering"
                    or expected_end_offset < 0
                    or (
                        has_pcm
                        and (
                            self._first_start_offset != 0
                            or self._last_range is None
                            or self._last_range[1] > expected_end_offset
                        )
                    )
                ):
                    self._discard_locked(reason="range_mismatch")
                    return False
                packets = tuple(self._packets)
                buffered_bytes = self._buffered_bytes
                self._state = "attaching"

            try:
                await self._bound_live_sink.start()
                if not self._owner_is_current():
                    with self._lock:
                        self._finish_revoke_locked()
                    return False
                with self._lock:
                    self._live_attached = True
                    if self._state != "attaching":
                        reset_live = True
                    else:
                        reset_live = False
                if reset_live:
                    self._reset_live()
                    with self._lock:
                        self._finish_revoke_locked()
                    return False

                self._reset_live()
                if before_playback is not None:
                    before_playback()
                await self._replay_packets(packets)
            except Exception:
                self._reset_live()
                with self._lock:
                    self._discard_locked(reason="provider_error")
                    self._live_attached = False
                logger.exception("Unable to promote private draft PCM")
                return False

            with self._lock:
                if self._state != "attaching":
                    self._reset_live()
                    self._finish_revoke_locked()
                    return False
                self._state = "promoted"
                self._packets.clear()
                self._buffered_bytes = 0
                # Keep actual received ranges. A text submission's end offset
                # does not say how much of its audio has arrived yet.
            self._emit(
                "voice_draft_tts_pcm_promoted",
                {"pcmBytes": buffered_bytes},
            )
            return True

    async def _replay_packets(
        self, packets: tuple[tuple[bytes, int, int], ...]
    ) -> None:
        """Coalesce packets of one sentence while preserving sentence ranges."""
        index = 0
        while index < len(packets):
            pcm, start, end = packets[index]
            index += 1
            if not pcm:
                await self._bound_live_sink.mark(end)
                continue
            parts = [pcm]
            while index < len(packets) and packets[index][1:] == (start, end):
                parts.append(packets[index][0])
                index += 1
            with self._lock:
                if self._state != "attaching":
                    return
            await self._bound_live_sink.append(
                b"".join(parts), start_offset=start, end_offset=end
            )

    async def revoke(self) -> None:
        with self._lock:
            if self._state == "discarded":
                return
            self._state = "revoking"
            discarded = self._buffered_bytes
            self._clear_locked()
            if discarded:
                self._emit(
                    "voice_draft_tts_pcm_discarded",
                    {"pcmBytes": discarded, "reason": "closed"},
                )
        async with self._io_gate:
            with self._lock:
                reset_live = self._live_attached
            if reset_live:
                self._reset_live()
            with self._lock:
                self._finish_revoke_locked()

    def discard(self) -> None:
        with self._lock:
            if self._state in {"attaching", "promoted", "revoking"}:
                self._state = "revoking"
                discarded = self._buffered_bytes
                self._clear_locked()
                if discarded:
                    self._emit(
                        "voice_draft_tts_pcm_discarded",
                        {"pcmBytes": discarded, "reason": "closed"},
                    )
            elif self._state != "discarded":
                self._discard_locked(reason="closed")

    async def stop(self) -> None:
        with self._lock:
            had_live_owner = self._state in {"attaching", "promoted"}
            live_sink = self._bound_live_sink
        await self.revoke()
        if had_live_owner:
            await live_sink.stop()

    async def pause(self) -> None:
        await self._run_promoted(self._bound_live_sink.pause)

    async def resume(self) -> None:
        await self._run_promoted(self._bound_live_sink.resume)

    async def duck(self) -> None:
        await self._run_promoted(self._bound_live_sink.duck)

    async def unduck(self) -> None:
        await self._run_promoted(self._bound_live_sink.unduck)

    async def wait_until_drained(self) -> None:
        # Do not hold the promotion mutation gate for the duration of physical
        # playback. A speech candidate must be able to pause the promoted sink
        # while this passive wait is still pending. Briefly acquire the gate as
        # a barrier against an append/promotion mutation that already started,
        # then retain the immutable owner-bound sink snapshot outside the gate.
        async with self._io_gate:
            with self._lock:
                if self._state != "promoted" or not self._live_attached:
                    return
                live_sink = self._bound_live_sink
        await live_sink.wait_until_drained()

    async def _run_promoted(
        self,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._io_gate:
            with self._lock:
                if self._state != "promoted":
                    return
            await operation()

    def _accept_range_locked(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
        enforce_cap: bool,
    ) -> bool:
        if (
            len(pcm) % 2
            or start_offset < 0
            or end_offset < start_offset
            or (
                enforce_cap and self._buffered_bytes + len(pcm) > self._max_buffer_bytes
            )
        ):
            return False
        current = (start_offset, end_offset)
        previous = self._last_range
        if previous is None:
            # After handoff the same provider may already have accepted more
            # public text before its first audio arrives. Its submitted-text
            # mapper owns the end bound; this sink enforces contiguous delivery.
            if start_offset != 0:
                return False
            self._first_start_offset = start_offset
        elif current != previous and start_offset != previous[1]:
            return False
        self._last_range = current
        return True

    def _discard_locked(self, *, reason: str) -> None:
        discarded = self._buffered_bytes
        self._state = "discarded"
        self._clear_locked()
        if discarded:
            self._emit(
                "voice_draft_tts_pcm_discarded",
                {"pcmBytes": discarded, "reason": reason},
            )

    def _finish_revoke_locked(self) -> None:
        self._state = "discarded"
        self._live_attached = False
        self._playback_owner = None
        self._bound_live_sink = self._live_sink
        self._clear_locked()

    def _reset_live(self) -> None:
        reset = getattr(self._bound_live_sink, "reset", None)
        if callable(reset):
            try:
                reset(0)
            except TypeError:
                reset()

    def _owner_is_current(self) -> bool:
        arbiter = self._playback_arbiter
        owner = self._playback_owner
        return arbiter is None or (owner is not None and arbiter.is_current(owner))

    def _clear_locked(self) -> None:
        self._packets.clear()
        self._buffered_bytes = 0
        self._first_start_offset = None
        self._last_range = None

    def _emit(self, name: str, metadata: dict[str, object]) -> None:
        if self._on_event is not None:
            self._on_event(name, metadata)
