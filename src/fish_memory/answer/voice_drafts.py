from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections import OrderedDict
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, TypeAlias, cast
from uuid import UUID

from fish_memory.agent.replies import AgentReplyContext, reply_context_fingerprint
from fish_memory.answer.output_sink import (
    AnswerOutputSnapshot,
    PromotableAnswerSink,
)
from fish_memory.answer.prepared_answer import PreparedAnswer
from fish_memory.answer.voice_draft_types import (
    VoiceDraftKey,
    VoiceDraftState,
    exact_content_hash,
)
from fish_memory.pet.types import PetEmotion

logger = logging.getLogger(__name__)

_EVENT_POLL_SECONDS = 0.005
_MAX_TOMBSTONES = 1024
_CONTEXT_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_NO_REPLY_CONTEXT_FINGERPRINT = reply_context_fingerprint(None)


class VoiceDraftRevisionConflict(ValueError):
    """Raised when one revision is reused for different candidate content."""


class VoiceDraftCoordinatorClosed(RuntimeError):
    """Raised when a voice commit starts after draft coordination closes."""


PrepareVoiceDraft: TypeAlias = Callable[
    [VoiceDraftKey, int, str],
    Awaitable[PreparedAnswer],
]


@dataclass(frozen=True, slots=True)
class VoiceDraftSnapshot:
    key: VoiceDraftKey
    revision: int
    content: str
    candidate_hash: str
    created_at: datetime
    state: VoiceDraftState
    output: AnswerOutputSnapshot
    ready: bool
    error: str | None = None
    context_fingerprint: str | None = None
    speech_started_at: datetime | None = None
    frame_snapshot_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class VoiceDraftEvent:
    event: str
    event_id: str
    data: dict[str, object]


@dataclass(frozen=True, slots=True)
class VoiceDraftLease:
    key: VoiceDraftKey
    revision: int
    candidate_hash: str
    input_hash: str
    created_at: datetime
    prepared: PreparedAnswer


@dataclass(frozen=True, slots=True)
class VoiceDraftTerminalMetrics:
    fish_id: UUID
    turn_id: UUID
    revision: int
    state: VoiceDraftState
    wasted_tokens: int | None


@dataclass(slots=True)
class _DraftEntry:
    key: VoiceDraftKey
    revision: int
    content: str
    candidate_hash: str
    created_at: datetime
    reply_context: AgentReplyContext | None = None
    reply_context_fingerprint: str = _NO_REPLY_CONTEXT_FINGERPRINT
    speech_started_at: datetime | None = None
    frame_snapshot_at: datetime | None = None
    state: VoiceDraftState = VoiceDraftState.ACTIVE
    output: AnswerOutputSnapshot = field(
        default_factory=lambda: AnswerOutputSnapshot(
            emotion=PetEmotion.NEUTRAL,
            end=False,
            content="",
            flush_offset=0,
        )
    )
    prepared: PreparedAnswer | None = None
    ready: bool = False
    error: str | None = None
    version: int = 0
    worker: asyncio.Task[None] | None = None
    expiry: asyncio.Task[None] | None = None
    prepared_available: asyncio.Event = field(default_factory=asyncio.Event)
    revoked: asyncio.Event = field(default_factory=asyncio.Event)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal_metrics_recorded: bool = False

    def snapshot(self) -> VoiceDraftSnapshot:
        return VoiceDraftSnapshot(
            key=self.key,
            revision=self.revision,
            content=self.content,
            candidate_hash=self.candidate_hash,
            created_at=self.created_at,
            state=self.state,
            output=self.output,
            ready=self.ready,
            error=self.error,
            context_fingerprint=self.reply_context_fingerprint,
            speech_started_at=self.speech_started_at,
            frame_snapshot_at=self.frame_snapshot_at,
        )


@dataclass(slots=True)
class _CommitGuardEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    references: int = 0


class VoiceDraftCoordinator:
    """Owns short-lived speculative answer preparation before durable commit."""

    def __init__(
        self,
        *,
        prepare: PrepareVoiceDraft,
        ttl_seconds: float = 2.0,
        max_concurrency: int = 2,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        terminal_metrics_sink: (
            Callable[[VoiceDraftTerminalMetrics], None] | None
        ) = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("voice draft TTL must be positive")
        if max_concurrency < 1:
            raise ValueError("voice draft concurrency must be positive")
        self._prepare = prepare
        self._prepare_accepts_created_at = _accepts_created_at(prepare)
        self._prepare_accepts_reply_context = _accepts_keyword(prepare, "reply_context")
        self._prepare_accepts_frame_window = all(
            _accepts_keyword(prepare, name)
            for name in ("speech_started_at", "frame_snapshot_at")
        )
        self._ttl_seconds = ttl_seconds
        self._max_concurrency = max_concurrency
        self._now = now
        self._terminal_metrics_sink = terminal_metrics_sink or _log_terminal_metrics
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._entries: dict[VoiceDraftKey, _DraftEntry] = {}
        self._tombstones: OrderedDict[VoiceDraftKey, VoiceDraftSnapshot] = OrderedDict()
        self._cancel_fences: OrderedDict[VoiceDraftKey, int] = OrderedDict()
        self._commit_guards: dict[VoiceDraftKey, _CommitGuardEntry] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @asynccontextmanager
    async def commit_guard(self, key: VoiceDraftKey) -> AsyncIterator[None]:
        """Serialize one turn's complete authoritative commit route."""

        async with self._lock:
            if self._closed:
                raise VoiceDraftCoordinatorClosed("voice draft coordinator is closed")
            entry = self._commit_guards.get(key)
            if entry is None:
                entry = _CommitGuardEntry()
                self._commit_guards[key] = entry
            entry.references += 1

        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            async with self._lock:
                if self._closed:
                    raise VoiceDraftCoordinatorClosed(
                        "voice draft coordinator is closed"
                    )
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._lock:
                entry.references -= 1
                if entry.references == 0 and self._commit_guards.get(key) is entry:
                    self._commit_guards.pop(key, None)

    async def upsert(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        reply_context: AgentReplyContext | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> VoiceDraftSnapshot | None:
        if revision < 1:
            raise ValueError("voice draft revision must be positive")
        if (speech_started_at is None) != (frame_snapshot_at is None):
            raise ValueError("voice frame timestamps must be supplied together")
        if speech_started_at is not None and frame_snapshot_at is not None:
            if (
                speech_started_at.utcoffset() is None
                or frame_snapshot_at.utcoffset() is None
                or speech_started_at > frame_snapshot_at
            ):
                raise ValueError("voice frame timestamps must be aware and ordered")
        candidate_hash = exact_content_hash(content)
        context_fingerprint = reply_context_fingerprint(reply_context)
        previous: _DraftEntry | None = None
        async with self._lock:
            if self._closed:
                return None
            cancel_fence = self._cancel_fences.get(key)
            if cancel_fence is not None and revision <= cancel_fence:
                raise VoiceDraftRevisionConflict(
                    "voice draft revision was terminally cancelled"
                )
            current = self._entries.get(key)
            if current is not None:
                if revision < current.revision:
                    return current.snapshot()
                if revision == current.revision:
                    if (
                        candidate_hash != current.candidate_hash
                        or context_fingerprint != current.reply_context_fingerprint
                        or speech_started_at != current.speech_started_at
                        or frame_snapshot_at != current.frame_snapshot_at
                    ):
                        raise VoiceDraftRevisionConflict(
                            "voice draft revision has different content "
                            "or reply context"
                        )
                    return current.snapshot()

            tombstone = self._tombstones.get(key)
            if tombstone is not None:
                if revision < tombstone.revision:
                    return tombstone
                if revision == tombstone.revision:
                    if (
                        candidate_hash != tombstone.candidate_hash
                        or context_fingerprint
                        != (
                            tombstone.context_fingerprint
                            or _NO_REPLY_CONTEXT_FINGERPRINT
                        )
                        or speech_started_at != tombstone.speech_started_at
                        or frame_snapshot_at != tombstone.frame_snapshot_at
                    ):
                        raise VoiceDraftRevisionConflict(
                            "voice draft revision has different content "
                            "or reply context"
                        )
                    return tombstone
                if tombstone.state is VoiceDraftState.COMMITTED:
                    return tombstone

            session_owner = self._session_owner_locked(key)
            if session_owner is None and len(self._entries) >= self._max_concurrency:
                return None
            if session_owner is not None:
                if session_owner.key != key or revision > session_owner.revision:
                    previous = session_owner
                    self._entries.pop(session_owner.key, None)
                    self._set_terminal_locked(
                        session_owner,
                        VoiceDraftState.CANCELLED,
                    )
                    self._record_tombstone_locked(session_owner)

            if tombstone is not None:
                self._tombstones.pop(key, None)
            if cancel_fence is not None:
                self._cancel_fences.pop(key, None)
            created_at = self._now()
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise ValueError("voice draft created_at must be timezone-aware")
            entry = _DraftEntry(
                key=key,
                revision=revision,
                content=content,
                candidate_hash=candidate_hash,
                created_at=created_at,
                reply_context=reply_context,
                reply_context_fingerprint=context_fingerprint,
                speech_started_at=speech_started_at,
                frame_snapshot_at=frame_snapshot_at,
            )
            self._entries[key] = entry
            entry.worker = self._start_task(
                self._run_prepare(entry),
                name=f"voice-draft-prepare:{key.turn_id}:{revision}",
            )
            entry.expiry = self._start_task(
                self._expire(entry),
                name=f"voice-draft-expiry:{key.turn_id}:{revision}",
            )
            snapshot = entry.snapshot()

        if previous is not None:
            await self._notify(previous)
            self._revoke(previous)
        return snapshot

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        state: VoiceDraftState = VoiceDraftState.CANCELLED,
    ) -> VoiceDraftSnapshot | None:
        if revision < 1:
            raise ValueError("voice draft revision must be positive")
        if state not in {VoiceDraftState.CANCELLED, VoiceDraftState.EXPIRED}:
            raise ValueError("voice draft cancel state must be cancelled or expired")
        entry: _DraftEntry | None = None
        async with self._lock:
            if self._closed:
                return None
            entry = self._entries.get(key)
            if entry is not None:
                if revision < entry.revision:
                    return None
                self._entries.pop(key, None)
                self._set_terminal_locked(entry, state)
                snapshot = entry.snapshot()
                if revision > entry.revision:
                    snapshot = replace(snapshot, revision=revision)
                self._record_tombstone_snapshot_locked(snapshot)
            else:
                tombstone = self._tombstones.get(key)
                if tombstone is not None:
                    if tombstone.state is VoiceDraftState.COMMITTED:
                        return tombstone
                    if revision <= tombstone.revision:
                        return tombstone
                    snapshot = replace(
                        tombstone,
                        revision=revision,
                        state=state,
                    )
                    self._record_tombstone_snapshot_locked(snapshot)
                else:
                    fence = self._cancel_fences.get(key)
                    if fence is None or revision > fence:
                        self._record_cancel_fence_locked(key, revision)
                    return None
        if entry is None:
            return snapshot
        await self._notify(entry)
        self._revoke(entry)
        return snapshot

    async def claim(
        self,
        key: VoiceDraftKey,
        *,
        promote_revision: int,
        content_hash: str,
    ) -> VoiceDraftLease | None:
        lease: VoiceDraftLease | None = None
        preserve_prepared = False
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.pop(key, None)
            prepared = entry.prepared
            input_hash = _available_input_hash(prepared)
            if (
                entry.revision == promote_revision
                and entry.candidate_hash == content_hash
                and prepared is not None
                and input_hash is not None
            ):
                self._set_terminal_locked(entry, VoiceDraftState.COMMITTED)
                preserve_prepared = True
                lease = VoiceDraftLease(
                    key=entry.key,
                    revision=entry.revision,
                    candidate_hash=entry.candidate_hash,
                    input_hash=input_hash,
                    created_at=entry.created_at,
                    prepared=prepared,
                )
            else:
                self._set_terminal_locked(entry, VoiceDraftState.CANCELLED)
            self._record_tombstone_locked(entry)
        cleanup = self._revoke(entry, preserve_prepared=preserve_prepared)
        if lease is None:
            self._schedule_notification(entry)
            return None
        try:
            await self._notify(entry)
            if cleanup is not None:
                await cleanup
        except BaseException:
            settlement: list[Awaitable[object]] = []
            if cleanup is not None:
                settlement.append(cleanup)
            if lease is not None:
                prepared_cleanup = self._cancel_prepared(lease.prepared)
                if prepared_cleanup is not None:
                    settlement.append(prepared_cleanup)
            await _settle_ignoring_cancellation(settlement)
            raise
        return lease

    async def finalize(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int | None,
        content: str,
        context_fingerprint: str | None = None,
    ) -> VoiceDraftLease | None:
        """Irreversibly fence a turn at its authoritative commit revision."""
        if (
            context_fingerprint is not None
            and _CONTEXT_FINGERPRINT_RE.fullmatch(context_fingerprint) is None
        ):
            raise ValueError(
                "voice reply context fingerprint must be a lowercase SHA-256"
            )
        if revision < 1:
            raise ValueError("voice turn commit revision must be positive")
        if (
            promote_revision is not None
            and promote_revision > 0
            and promote_revision >= revision
        ):
            raise ValueError("promote revision must be earlier than commit revision")
        final_hash = exact_content_hash(content)
        awaited_entry = await self._await_matching_prepared(
            key,
            promote_revision=promote_revision,
            content_hash=final_hash,
            context_fingerprint=context_fingerprint,
        )
        lease: VoiceDraftLease | None = None
        preserve_prepared = False
        entry: _DraftEntry | None = None
        async with self._lock:
            if self._closed:
                raise VoiceDraftCoordinatorClosed("voice draft coordinator is closed")
            existing = self._tombstones.get(key)
            if existing is not None and existing.state is VoiceDraftState.COMMITTED:
                if (
                    existing.candidate_hash != final_hash
                    or existing.context_fingerprint != context_fingerprint
                ):
                    raise VoiceDraftRevisionConflict(
                        "voice turn commit has different content or reply context"
                    )
                return None
            if (
                awaited_entry is not None
                and not self._owns_locked(awaited_entry)
                and (awaited_entry.error is None or key in self._entries)
            ):
                raise VoiceDraftRevisionConflict(
                    "voice draft owner changed while its preparation was pending"
                )
            entry = self._entries.pop(key, None)
            if entry is not None:
                prepared = entry.prepared
                input_hash = _available_input_hash(prepared)
                if (
                    promote_revision is not None
                    and promote_revision > 0
                    and entry.revision == promote_revision
                    and entry.candidate_hash == final_hash
                    and _reply_context_matches(entry, context_fingerprint)
                    and prepared is not None
                    and input_hash is not None
                ):
                    self._set_terminal_locked(entry, VoiceDraftState.COMMITTED)
                    preserve_prepared = True
                    lease = VoiceDraftLease(
                        key=entry.key,
                        revision=entry.revision,
                        candidate_hash=entry.candidate_hash,
                        input_hash=input_hash,
                        created_at=entry.created_at,
                        prepared=prepared,
                    )
                else:
                    self._set_terminal_locked(entry, VoiceDraftState.CANCELLED)
            created_at = entry.created_at if entry is not None else self._now()
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise ValueError("voice draft created_at must be timezone-aware")
            authoritative = VoiceDraftSnapshot(
                key=key,
                revision=revision,
                content=content,
                candidate_hash=final_hash,
                created_at=created_at,
                state=VoiceDraftState.COMMITTED,
                output=(
                    entry.output
                    if entry is not None
                    else AnswerOutputSnapshot(
                        emotion=PetEmotion.NEUTRAL,
                        end=False,
                        content="",
                        flush_offset=0,
                    )
                ),
                ready=entry.ready if entry is not None else False,
                context_fingerprint=context_fingerprint,
            )
            self._record_tombstone_snapshot_locked(authoritative)

        if entry is None:
            return None
        cleanup = self._revoke(entry, preserve_prepared=preserve_prepared)
        if lease is None:
            self._schedule_notification(entry)
            return None
        try:
            await self._notify(entry)
            if cleanup is not None:
                await cleanup
        except BaseException:
            settlement: list[Awaitable[object]] = []
            if cleanup is not None:
                settlement.append(cleanup)
            prepared_cleanup = self._cancel_prepared(lease.prepared)
            if prepared_cleanup is not None:
                settlement.append(prepared_cleanup)
            await _settle_ignoring_cancellation(settlement)
            raise
        return lease

    async def _await_matching_prepared(
        self,
        key: VoiceDraftKey,
        *,
        promote_revision: int | None,
        content_hash: str,
        context_fingerprint: str | None,
    ) -> _DraftEntry | None:
        """Take over the existing context load without holding the owner lock."""
        async with self._lock:
            entry = self._entries.get(key)
            if (
                entry is None
                or promote_revision is None
                or promote_revision <= 0
                or entry.revision != promote_revision
                or entry.candidate_hash != content_hash
                or not _reply_context_matches(entry, context_fingerprint)
            ):
                return None
            # An authoritative handoff has no speculative TTL. Keep the entry
            # active so resumed speech, a newer owner, or close can still revoke it.
            if entry.expiry is not None:
                entry.expiry.cancel()

        try:
            await entry.prepared_available.wait()
            if entry.prepared is not None and not entry.revoked.is_set():
                await self._await_input_ready(entry, entry.prepared)
        except asyncio.CancelledError:
            async with self._lock:
                revoke = self._owns_locked(entry)
                if revoke:
                    self._entries.pop(entry.key, None)
                    self._set_terminal_locked(entry, VoiceDraftState.CANCELLED)
                    self._record_tombstone_locked(entry)
            if revoke:
                self._schedule_notification(entry)
                self._revoke(entry)
            raise
        return entry

    @staticmethod
    async def _await_input_ready(
        entry: _DraftEntry,
        prepared: PreparedAnswer,
    ) -> None:
        if _available_input_hash(prepared) is not None:
            return
        input_ready = getattr(prepared, "input_ready", None)
        if not callable(input_ready):
            return
        ready = asyncio.create_task(input_ready())
        revoked = asyncio.create_task(entry.revoked.wait())
        try:
            await asyncio.wait((ready, revoked), return_when=asyncio.FIRST_COMPLETED)
            if entry.revoked.is_set():
                return
            if ready.done():
                try:
                    await ready
                except Exception:
                    # Failed preparation has no reusable audit hash. Finalize
                    # revokes it and lets the authoritative answer take over.
                    return
        finally:
            for task in (ready, revoked):
                if not task.done():
                    task.cancel()
            await asyncio.gather(ready, revoked, return_exceptions=True)

    async def iter_events(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
    ) -> AsyncGenerator[VoiceDraftEvent, None]:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.revision != revision:
                return
        observed = entry.snapshot()
        observed_version = entry.version
        yield _snapshot_event(observed, observed_version)

        while observed.state is VoiceDraftState.ACTIVE:
            async with entry.condition:
                await entry.condition.wait_for(
                    lambda version=observed_version: entry.version != version
                )
                current = entry.snapshot()
                current_version = entry.version

            events = _changed_events(observed, current, current_version)
            observed = current
            observed_version = current_version
            for event in events:
                yield event
            if current.state is not VoiceDraftState.ACTIVE:
                return

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                entries: list[_DraftEntry] = []
            else:
                self._closed = True
                entries = list(self._entries.values())
                self._entries.clear()
                for entry in entries:
                    self._set_terminal_locked(entry, VoiceDraftState.CANCELLED)
                    self._record_tombstone_locked(entry)
        for entry in entries:
            await self._notify(entry)
            self._revoke(entry)
        await self._drain_owned_tasks()

    async def _run_prepare(self, entry: _DraftEntry) -> None:
        prepared: PreparedAnswer | None = None
        try:
            async with self._semaphore:
                prepared = await self._call_prepare(entry)
                async with self._lock:
                    if not self._owns_locked(entry):
                        stale = True
                    else:
                        stale = False
                        entry.prepared = prepared
                        entry.prepared_available.set()
                        entry.version += 1
                if stale:
                    if entry.state is not VoiceDraftState.COMMITTED:
                        entry.prepared = prepared
                        cleanup = self._cancel_prepared(prepared)
                        if cleanup is not None:
                            self._schedule_cleanup(entry, [cleanup])
                        else:
                            self._record_terminal_metrics(entry)
                    return
                await self._observe_output(entry, prepared)
                await self._notify(entry)

                preparation_done = getattr(
                    prepared, "preparation_done", prepared.initial_done
                )
                preparation_ready = getattr(
                    prepared, "preparation_ready", prepared.initial_ready
                )
                while not preparation_done():
                    await asyncio.sleep(_EVENT_POLL_SECONDS)
                    await self._observe_output(entry, prepared)
                await preparation_ready()
                await self._observe_output(entry, prepared)
                async with self._lock:
                    if not self._owns_locked(entry):
                        return
                    entry.ready = True
                    entry.version += 1
                await self._notify(entry)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "Voice draft preparation failed for turn %s revision %d",
                entry.key.turn_id,
                entry.revision,
            )
            async with self._lock:
                if not self._owns_locked(entry):
                    return
                self._entries.pop(entry.key, None)
                entry.error = type(error).__name__
                self._set_terminal_locked(entry, VoiceDraftState.CANCELLED)
                self._record_tombstone_locked(entry)
            await self._notify(entry)
            if prepared is not None:
                entry.prepared = prepared
                cleanup = self._cancel_prepared(prepared)
                if cleanup is not None:
                    self._schedule_cleanup(entry, [cleanup])
                else:
                    self._record_terminal_metrics(entry)
            else:
                self._record_terminal_metrics(entry)

    async def _call_prepare(self, entry: _DraftEntry) -> PreparedAnswer:
        keyword_args: dict[str, object] = {}
        if self._prepare_accepts_created_at:
            keyword_args["created_at"] = entry.created_at
        if self._prepare_accepts_reply_context:
            keyword_args["reply_context"] = entry.reply_context
        if self._prepare_accepts_frame_window and entry.speech_started_at is not None:
            keyword_args["speech_started_at"] = entry.speech_started_at
            keyword_args["frame_snapshot_at"] = entry.frame_snapshot_at
        if keyword_args:
            prepare = cast(
                Callable[..., Awaitable[PreparedAnswer]],
                self._prepare,
            )
            return await prepare(
                entry.key,
                entry.revision,
                entry.content,
                **keyword_args,
            )
        return await self._prepare(entry.key, entry.revision, entry.content)

    async def _observe_output(
        self,
        entry: _DraftEntry,
        prepared: PreparedAnswer,
    ) -> None:
        sink = prepared.sink
        if not isinstance(sink, PromotableAnswerSink):
            raise TypeError("voice draft preparation requires a promotable sink")
        output = sink.snapshot()
        async with self._lock:
            if not self._owns_locked(entry) or output == entry.output:
                return
            entry.output = output
            entry.version += 1
        await self._notify(entry)

    async def _expire(self, entry: _DraftEntry) -> None:
        try:
            await asyncio.sleep(self._ttl_seconds)
            async with self._lock:
                if not self._owns_locked(entry):
                    return
                self._entries.pop(entry.key, None)
                self._set_terminal_locked(entry, VoiceDraftState.EXPIRED)
                self._record_tombstone_locked(entry)
            await self._notify(entry)
            self._revoke(entry)
        except asyncio.CancelledError:
            raise

    def _session_owner_locked(self, key: VoiceDraftKey) -> _DraftEntry | None:
        for entry in self._entries.values():
            if (
                entry.key.fish_id == key.fish_id
                and entry.key.session_id == key.session_id
            ):
                return entry
        return None

    def _owns_locked(self, entry: _DraftEntry) -> bool:
        current = self._entries.get(entry.key)
        return current is entry and current.revision == entry.revision

    def _record_tombstone_locked(self, entry: _DraftEntry) -> None:
        self._record_tombstone_snapshot_locked(entry.snapshot())

    def _record_tombstone_snapshot_locked(
        self,
        snapshot: VoiceDraftSnapshot,
    ) -> None:
        existing = self._tombstones.get(snapshot.key)
        if existing is not None:
            if existing.state is VoiceDraftState.COMMITTED:
                return
            if (
                snapshot.state is not VoiceDraftState.COMMITTED
                and existing.revision > snapshot.revision
            ):
                return
        self._cancel_fences.pop(snapshot.key, None)
        self._tombstones[snapshot.key] = snapshot
        self._tombstones.move_to_end(snapshot.key)
        while len(self._tombstones) > _MAX_TOMBSTONES:
            self._tombstones.popitem(last=False)

    def _record_cancel_fence_locked(self, key: VoiceDraftKey, revision: int) -> None:
        self._cancel_fences[key] = revision
        self._cancel_fences.move_to_end(key)
        while len(self._cancel_fences) > _MAX_TOMBSTONES:
            self._cancel_fences.popitem(last=False)

    @staticmethod
    def _set_terminal_locked(
        entry: _DraftEntry,
        state: VoiceDraftState,
    ) -> None:
        entry.state = state
        entry.prepared_available.set()
        entry.revoked.set()
        entry.version += 1

    @staticmethod
    async def _notify(entry: _DraftEntry) -> None:
        async with entry.condition:
            entry.condition.notify_all()

    def _revoke(
        self,
        entry: _DraftEntry,
        *,
        preserve_prepared: bool = False,
    ) -> asyncio.Task[None] | None:
        current = asyncio.current_task()
        tasks: list[Awaitable[object]] = []
        if entry.expiry is not None and entry.expiry is not current:
            entry.expiry.cancel()
            tasks.append(entry.expiry)
        if entry.worker is not None and entry.worker is not current:
            entry.worker.cancel()
            tasks.append(entry.worker)
        if not preserve_prepared and entry.prepared is not None:
            cleanup = self._cancel_prepared(entry.prepared)
            if cleanup is not None:
                tasks.append(cleanup)
        if tasks:
            return self._schedule_cleanup(entry, tasks)
        self._record_terminal_metrics(entry)
        return None

    def _schedule_cleanup(
        self,
        entry: _DraftEntry,
        tasks: list[Awaitable[object]],
    ) -> asyncio.Task[None]:
        cleanup_task = asyncio.create_task(
            self._settle_and_record_terminal(entry, tasks),
            name=f"voice-draft-cleanup:{entry.key.turn_id}:{entry.revision}",
        )
        self._cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._cleanup_tasks.discard)
        cleanup_task.add_done_callback(_consume_task)
        return cleanup_task

    def _schedule_notification(self, entry: _DraftEntry) -> None:
        task = asyncio.create_task(
            self._notify(entry),
            name=f"voice-draft-notify:{entry.key.turn_id}:{entry.revision}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        task.add_done_callback(_consume_task)

    async def _settle_and_record_terminal(
        self,
        entry: _DraftEntry,
        tasks: list[Awaitable[object]],
    ) -> None:
        await _settle(tasks)
        self._record_terminal_metrics(entry)

    def _record_terminal_metrics(self, entry: _DraftEntry) -> None:
        if entry.terminal_metrics_recorded or entry.state in {
            VoiceDraftState.ACTIVE,
            VoiceDraftState.COMMITTED,
        }:
            return
        entry.terminal_metrics_recorded = True
        usage = _prepared_output_token_usage(entry.prepared)
        try:
            metrics = VoiceDraftTerminalMetrics(
                fish_id=entry.key.fish_id,
                turn_id=entry.key.turn_id,
                revision=entry.revision,
                state=entry.state,
                wasted_tokens=usage,
            )
            self._terminal_metrics_sink(metrics)
        except Exception:
            logger.exception("Unable to record voice draft terminal metrics")

    async def _drain_owned_tasks(self) -> None:
        current = asyncio.current_task()
        while True:
            owned = tuple(
                task for task in self._tasks if task is not current and not task.done()
            )
            for task in owned:
                task.cancel()
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)

            cleanup = tuple(task for task in self._cleanup_tasks if not task.done())
            if cleanup:
                await asyncio.gather(*cleanup, return_exceptions=True)

            if not any(
                task is not current and not task.done() for task in self._tasks
            ) and not any(not task.done() for task in self._cleanup_tasks):
                return

    @staticmethod
    def _cancel_prepared(
        prepared: PreparedAnswer,
    ) -> Awaitable[object] | None:
        try:
            cleanup = prepared.cancel()
        except Exception:
            logger.exception("Failed to cancel prepared voice draft")
            return None
        return cleanup

    def _start_task(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_consume_task)
        return task


def _accepts_created_at(prepare: PrepareVoiceDraft) -> bool:
    return _accepts_keyword(prepare, "created_at")


def _accepts_keyword(prepare: PrepareVoiceDraft, keyword: str) -> bool:
    try:
        return keyword in inspect.signature(prepare).parameters
    except (TypeError, ValueError):
        return False


def _reply_context_matches(entry: _DraftEntry, context_fingerprint: str | None) -> bool:
    return entry.reply_context_fingerprint == (
        context_fingerprint or _NO_REPLY_CONTEXT_FINGERPRINT
    )


def _available_input_hash(prepared: PreparedAnswer | None) -> str | None:
    if prepared is None:
        return None
    try:
        return prepared.input_hash
    except RuntimeError:
        return None


def _log_terminal_metrics(metrics: VoiceDraftTerminalMetrics) -> None:
    logger.info(
        "Voice draft terminal metrics: turn_id=%s revision=%d state=%s "
        "wasted_tokens=%s",
        metrics.turn_id,
        metrics.revision,
        metrics.state.value,
        metrics.wasted_tokens,
    )


def _prepared_output_token_usage(
    prepared: PreparedAnswer | None,
) -> int | None:
    if prepared is None:
        return None
    usage = getattr(prepared, "output_token_usage", None)
    if not callable(usage):
        return None
    value = usage()
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _snapshot_event(
    snapshot: VoiceDraftSnapshot,
    version: int,
) -> VoiceDraftEvent:
    return VoiceDraftEvent(
        event="draft.snapshot",
        event_id=f"{snapshot.key.turn_id}:{snapshot.revision}:snapshot:{version}",
        data=_event_data(snapshot),
    )


def _changed_events(
    previous: VoiceDraftSnapshot,
    current: VoiceDraftSnapshot,
    version: int,
) -> list[VoiceDraftEvent]:
    events: list[VoiceDraftEvent] = []
    terminal = current.state is not VoiceDraftState.ACTIVE
    output_changed = (
        previous.output.content != current.output.content
        or previous.output.speech_content != current.output.speech_content
        or previous.output.emotion != current.output.emotion
        or previous.output.end != current.output.end
    )
    if terminal:
        event_name = (
            "draft.failed"
            if current.error is not None
            else (
                "draft.canceled"
                if current.state in {VoiceDraftState.CANCELLED, VoiceDraftState.EXPIRED}
                else "draft.snapshot"
            )
        )
        events.append(
            VoiceDraftEvent(
                event=event_name,
                event_id=(
                    f"{current.key.turn_id}:{current.revision}:terminal:{version}"
                ),
                data=_event_data(current),
            )
        )
        return events
    if output_changed:
        events.append(_snapshot_event(current, version))
    if previous.output.flush_offset != current.output.flush_offset:
        events.append(
            VoiceDraftEvent(
                event="draft.segment",
                event_id=(
                    f"{current.key.turn_id}:{current.revision}:"
                    f"flush:{current.output.flush_offset}"
                ),
                data={
                    "turnId": str(current.key.turn_id),
                    "revision": current.revision,
                    "flushOffset": current.output.flush_offset,
                    "reason": "lane_answer_complete",
                },
            )
        )
    if not previous.ready and current.ready:
        events.append(
            VoiceDraftEvent(
                event="draft.ready",
                event_id=f"{current.key.turn_id}:{current.revision}:ready:{version}",
                data=_event_data(current),
            )
        )
    return events


def _event_data(snapshot: VoiceDraftSnapshot) -> dict[str, object]:
    return {
        "turnId": str(snapshot.key.turn_id),
        "revision": snapshot.revision,
        "state": snapshot.state.value,
        "content": snapshot.output.content,
        "end": snapshot.output.end,
        "emotion": snapshot.output.emotion.value,
        "startOffset": 0,
        "endOffset": len(snapshot.output.content),
        "flushOffset": snapshot.output.flush_offset,
        **(
            {"speechContent": snapshot.output.speech_content, "speechTags": True}
            if snapshot.output.speech_tags
            else {}
        ),
    }


async def _settle(tasks: list[Awaitable[object]]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


async def _settle_ignoring_cancellation(tasks: list[Awaitable[object]]) -> None:
    if not tasks:
        return
    settlement = asyncio.create_task(_settle(tasks))
    while not settlement.done():
        try:
            await asyncio.shield(settlement)
        except asyncio.CancelledError:
            continue
    await settlement


def _consume_task(task: asyncio.Task[object]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass
