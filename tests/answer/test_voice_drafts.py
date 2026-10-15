import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest

from fish_memory.agent.models import AgentMessageSubjectKind, AgentResolvedTarget
from fish_memory.agent.replies import (
    AgentReplyContext,
    AgentReplyDisposition,
    AgentReplySubjectSnapshot,
    PreparedAgentReply,
    RunResultReplyAction,
    reply_context_fingerprint,
)
from fish_memory.agent.routing import TaskLaneContext
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.voice_draft_types import (
    VoiceDraftKey,
    VoiceDraftState,
    exact_content_hash,
)
from fish_memory.answer.voice_drafts import (
    PrepareVoiceDraft,
    VoiceDraftCoordinator,
    VoiceDraftCoordinatorClosed,
    VoiceDraftRevisionConflict,
    VoiceDraftSnapshot,
    VoiceDraftTerminalMetrics,
)
from fish_memory.app import _voice_draft_preparer

FISH_ID = UUID("00000000-0000-0000-0000-000000000101")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000102")
TURN_ID = UUID("00000000-0000-0000-0000-000000000103")
NOW = datetime(2026, 8, 30, 1, 30, tzinfo=UTC)


class FakePrepared:
    def __init__(
        self,
        revision: int,
        *,
        input_hash_ready: bool = True,
        output_tokens: int | None = None,
    ) -> None:
        self.revision = revision
        self.sink = PromotableAnswerSink()
        self.ready = asyncio.Event()
        self.cancelled = False
        self._input_hash_ready = input_hash_ready
        self._output_tokens = output_tokens

    @property
    def input_hash(self) -> str:
        if not self._input_hash_ready:
            raise RuntimeError("input hash is not ready")
        return f"input:{self.revision}"

    async def initial_ready(self) -> None:
        await self.ready.wait()

    def initial_done(self) -> bool:
        return self.ready.is_set()

    def cancel(self) -> asyncio.Task[None]:
        self.cancelled = True
        self.ready.set()
        return asyncio.create_task(asyncio.sleep(0))

    def output_token_usage(self) -> int | None:
        return self._output_tokens


class BlockingSettlementPrepared(FakePrepared):
    def __init__(self, revision: int) -> None:
        super().__init__(revision)
        self.cancel_started = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.settled = asyncio.Event()
        self._cleanup_task: asyncio.Task[None] | None = None

    def cancel(self) -> asyncio.Task[None]:
        self.cancelled = True
        self.ready.set()
        self.cancel_started.set()
        if self._cleanup_task is None:

            async def settle() -> None:
                await self.cancel_release.wait()
                self.settled.set()

            self._cleanup_task = asyncio.create_task(settle())
        return self._cleanup_task


class BlockingPreparer:
    def __init__(self, *, suppress_cancel: bool = False) -> None:
        self.items: list[FakePrepared] = []
        self.release: list[asyncio.Event] = []
        self.suppress_cancel = suppress_cancel

    async def __call__(
        self,
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del key, content
        item = FakePrepared(revision)
        gate = asyncio.Event()
        self.items.append(item)
        self.release.append(gate)
        try:
            await gate.wait()
        except asyncio.CancelledError:
            if not self.suppress_cancel:
                raise
            await gate.wait()
        item.ready.set()
        return item


class ImmediatePreparer:
    def __init__(
        self,
        *,
        input_hash_ready: bool = True,
        output_tokens: int | None = None,
    ) -> None:
        self.items: list[FakePrepared] = []
        self.input_hash_ready = input_hash_ready
        self.output_tokens = output_tokens

    async def __call__(
        self,
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del key, content
        item = FakePrepared(
            revision,
            input_hash_ready=self.input_hash_ready,
            output_tokens=self.output_tokens,
        )
        self.items.append(item)
        return item


async def wait_until(
    predicate: Callable[[], bool],
    *,
    deadline_seconds: float = 0.5,
) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=deadline_seconds)


def key(
    *,
    session_id: UUID = SESSION_ID,
    turn_id: UUID = TURN_ID,
) -> VoiceDraftKey:
    return VoiceDraftKey(FISH_ID, session_id, turn_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_tokens", [7, None])
async def test_canceled_draft_reports_only_actual_wasted_provider_tokens(
    output_tokens: int | None,
) -> None:
    metrics: list[VoiceDraftTerminalMetrics] = []
    preparer = ImmediatePreparer(output_tokens=output_tokens)
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=2,
        max_concurrency=2,
        now=lambda: NOW,
        terminal_metrics_sink=metrics.append,
    )
    draft_key = key()

    await coordinator.upsert(draft_key, revision=1, content="候选问题")
    await wait_until(lambda: bool(preparer.items))
    preparer.items[0].ready.set()
    await wait_until(
        lambda: bool(
            coordinator._entries.get(draft_key)
            and coordinator._entries[draft_key].ready
        )
    )
    await coordinator.cancel(draft_key, revision=2)
    await coordinator.close()

    assert metrics == [
        VoiceDraftTerminalMetrics(
            fish_id=FISH_ID,
            turn_id=TURN_ID,
            revision=1,
            state=VoiceDraftState.CANCELLED,
            wasted_tokens=output_tokens,
        )
    ]


@pytest.mark.asyncio
async def test_higher_revision_revokes_late_draft_completion() -> None:
    preparer = BlockingPreparer(suppress_cancel=True)
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()

    await coordinator.upsert(draft_key, revision=1, content="前半句")
    await wait_until(lambda: len(preparer.items) == 1)
    await coordinator.upsert(draft_key, revision=3, content="完整问题")
    await wait_until(lambda: len(preparer.items) == 2)

    first, second = preparer.items
    preparer.release[0].set()
    preparer.release[1].set()
    await asyncio.sleep(0)
    lease = await coordinator.claim(
        draft_key,
        promote_revision=3,
        content_hash=exact_content_hash("完整问题"),
    )

    assert lease is not None
    assert lease.revision == 3
    assert lease.prepared is second
    assert first.cancelled
    await coordinator.close()


@pytest.mark.asyncio
async def test_revision_cas_is_idempotent_and_preserves_created_at() -> None:
    created_times = iter((NOW, NOW + timedelta(seconds=1)))
    preparer = BlockingPreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: next(created_times),
    )
    draft_key = key()

    first = await coordinator.upsert(draft_key, revision=3, content="问题")
    duplicate = await coordinator.upsert(draft_key, revision=3, content="问题")
    lower = await coordinator.upsert(draft_key, revision=2, content="旧问题")
    with pytest.raises(VoiceDraftRevisionConflict):
        await coordinator.upsert(draft_key, revision=3, content="不同问题")

    assert first is not None
    assert duplicate == first
    assert lower == first
    assert duplicate.created_at == NOW
    await wait_until(lambda: len(preparer.items) == 1)
    await coordinator.close()


@pytest.mark.asyncio
async def test_one_active_turn_per_session_replaces_the_previous_owner() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    first_key = key()
    second_key = key(turn_id=UUID("00000000-0000-0000-0000-000000000104"))

    await coordinator.upsert(first_key, revision=1, content="first")
    await wait_until(lambda: len(preparer.items) == 1)
    await coordinator.upsert(second_key, revision=1, content="second")
    await wait_until(lambda: len(preparer.items) == 2)

    assert preparer.items[0].cancelled
    assert (
        await coordinator.claim(
            first_key,
            promote_revision=1,
            content_hash=exact_content_hash("first"),
        )
        is None
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_capacity_miss_returns_without_waiting() -> None:
    preparer = BlockingPreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    first_key = key()
    second_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000112"),
        turn_id=UUID("00000000-0000-0000-0000-000000000113"),
    )
    third_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000122"),
        turn_id=UUID("00000000-0000-0000-0000-000000000123"),
    )
    await coordinator.upsert(first_key, revision=1, content="one")
    await coordinator.upsert(second_key, revision=1, content="two")
    await wait_until(lambda: len(preparer.items) == 2)

    missed = await asyncio.wait_for(
        coordinator.upsert(third_key, revision=1, content="three"),
        timeout=0.02,
    )

    assert missed is None
    assert len(preparer.items) == 2
    await coordinator.close()


@pytest.mark.asyncio
async def test_claim_never_waits_for_initial_ready() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)

    lease = await asyncio.wait_for(
        coordinator.claim(
            draft_key,
            promote_revision=1,
            content_hash=exact_content_hash("question"),
        ),
        timeout=0.02,
    )

    assert lease is not None
    assert lease.input_hash == "input:1"
    assert lease.prepared is preparer.items[0]
    assert not preparer.items[0].ready.is_set()
    await coordinator.close()


@pytest.mark.asyncio
async def test_cancelled_claim_settles_prepared_owner_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = BlockingSettlementPrepared(1)

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del draft_key, revision, content
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(
        lambda: coordinator._entries[draft_key].prepared is prepared,
    )
    notify_started = asyncio.Event()

    async def blocked_notify(entry: object) -> None:
        del entry
        notify_started.set()
        await asyncio.Future()

    monkeypatch.setattr(coordinator, "_notify", blocked_notify)
    claim_task = asyncio.create_task(
        coordinator.claim(
            draft_key,
            promote_revision=1,
            content_hash=exact_content_hash("question"),
        )
    )
    try:
        await asyncio.wait_for(notify_started.wait(), timeout=0.2)
        claim_task.cancel()
        await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)

        assert not claim_task.done()
        prepared.cancel_release.set()
        with pytest.raises(asyncio.CancelledError):
            await claim_task
        assert prepared.settled.is_set()
    finally:
        prepared.cancel_release.set()
        await asyncio.gather(claim_task, return_exceptions=True)
        if not prepared.cancelled:
            await prepared.cancel()
        await coordinator.close()


@pytest.mark.asyncio
async def test_claim_detaches_observers_and_releases_capacity() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    first_key = key()
    second_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000132"),
        turn_id=UUID("00000000-0000-0000-0000-000000000133"),
    )
    third_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000142"),
        turn_id=UUID("00000000-0000-0000-0000-000000000143"),
    )
    await coordinator.upsert(first_key, revision=1, content="one")
    await coordinator.upsert(second_key, revision=1, content="two")
    await wait_until(lambda: len(preparer.items) == 2)

    first_lease = await coordinator.claim(
        first_key,
        promote_revision=1,
        content_hash=exact_content_hash("one"),
    )
    second_lease = await coordinator.claim(
        second_key,
        promote_revision=1,
        content_hash=exact_content_hash("two"),
    )
    third = await coordinator.upsert(third_key, revision=1, content="three")
    await wait_until(lambda: len(preparer.items) == 3)

    assert first_lease is not None
    assert second_lease is not None
    assert third is not None
    assert not preparer.items[0].cancelled
    assert not preparer.items[1].cancelled
    await coordinator.close()
    await asyncio.sleep(0)
    current = asyncio.current_task()
    leaked = [
        task
        for task in asyncio.all_tasks()
        if task is not current
        and not task.done()
        and task.get_name().startswith("voice-draft-")
    ]
    assert leaked == []


@pytest.mark.asyncio
async def test_claim_miss_revokes_unattachable_prepared_work_immediately() -> None:
    preparer = ImmediatePreparer(input_hash_ready=False)
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)

    lease = await asyncio.wait_for(
        coordinator.claim(
            draft_key,
            promote_revision=1,
            content_hash=exact_content_hash("question"),
        ),
        timeout=0.02,
    )

    assert lease is None
    await wait_until(lambda: preparer.items[0].cancelled)
    await coordinator.close()


@pytest.mark.asyncio
async def test_finalize_miss_returns_before_blocking_prepared_cleanup() -> None:
    prepared = BlockingSettlementPrepared(1)

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del draft_key, revision, content
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="draft")
    await wait_until(lambda: coordinator._entries[draft_key].prepared is prepared)

    lease = await asyncio.wait_for(
        coordinator.finalize(
            draft_key,
            revision=2,
            promote_revision=1,
            content="authoritative",
        ),
        timeout=0.02,
    )

    assert lease is None
    await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
    assert not prepared.settled.is_set()
    close_task = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    prepared.cancel_release.set()
    await asyncio.wait_for(close_task, timeout=0.2)
    assert prepared.settled.is_set()


@pytest.mark.asyncio
async def test_finalize_adopts_the_same_pending_context_past_draft_ttl() -> None:
    preparer = BlockingPreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=0.02,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    commit = asyncio.create_task(
        coordinator.finalize(
            draft_key, revision=2, promote_revision=1, content="question"
        )
    )
    try:
        await asyncio.sleep(0.04)
        assert not commit.done()
        assert coordinator._entries[draft_key].state is VoiceDraftState.ACTIVE
        preparer.release[0].set()
        lease = await asyncio.wait_for(commit, timeout=0.2)

        assert lease is not None
        assert lease.prepared is preparer.items[0]
        assert len(preparer.items) == 1
        assert not preparer.items[0].cancelled
        await lease.prepared.cancel()
    finally:
        commit.cancel()
        await asyncio.gather(commit, return_exceptions=True)
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["cancel", "revision", "owner", "close"])
async def test_finalize_pending_context_is_revoked_without_a_fallback(
    invalidate: str,
) -> None:
    preparer = BlockingPreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    commit = asyncio.create_task(
        coordinator.finalize(
            draft_key, revision=2, promote_revision=1, content="question"
        )
    )
    await asyncio.sleep(0)
    try:
        if invalidate == "cancel":
            await coordinator.cancel(draft_key, revision=2)
        elif invalidate == "revision":
            await coordinator.upsert(draft_key, revision=3, content="continued")
        elif invalidate == "owner":
            await coordinator.upsert(
                key(turn_id=UUID(int=999)), revision=1, content="new owner"
            )
        else:
            await coordinator.close()

        error = (
            VoiceDraftCoordinatorClosed
            if invalidate == "close"
            else VoiceDraftRevisionConflict
        )
        with pytest.raises(error):
            await asyncio.wait_for(commit, timeout=0.2)
        assert coordinator._tombstones[draft_key].state is VoiceDraftState.CANCELLED
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_cancelled_finalize_revokes_the_pending_context_load() -> None:
    preparer = BlockingPreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer), now=lambda: NOW
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    worker = coordinator._entries[draft_key].worker
    commit = asyncio.create_task(
        coordinator.finalize(
            draft_key, revision=2, promote_revision=1, content="question"
        )
    )
    await asyncio.sleep(0)
    commit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit
    await coordinator.close()

    assert worker is not None and worker.cancelled()
    assert draft_key not in coordinator._entries
    assert coordinator._tombstones[draft_key].state is VoiceDraftState.CANCELLED


@pytest.mark.asyncio
async def test_pending_preparation_failure_releases_authoritative_fallback() -> None:
    started = asyncio.Event()
    fail = asyncio.Event()

    async def prepare(*args: object) -> FakePrepared:
        del args
        started.set()
        await fail.wait()
        raise RuntimeError("context unavailable")

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare), now=lambda: NOW
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await started.wait()
    commit = asyncio.create_task(
        coordinator.finalize(
            draft_key, revision=2, promote_revision=1, content="question"
        )
    )
    try:
        await asyncio.sleep(0)
        fail.set()
        assert await asyncio.wait_for(commit, timeout=0.2) is None
        assert coordinator._tombstones[draft_key].state is VoiceDraftState.COMMITTED
    finally:
        commit.cancel()
        await asyncio.gather(commit, return_exceptions=True)
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", [False, True])
async def test_finalize_waits_for_existing_input_audit_only(revoke: bool) -> None:
    class HashPendingPrepared(FakePrepared):
        def __init__(self) -> None:
            super().__init__(1, input_hash_ready=False)
            self.hash_release = asyncio.Event()
            self.hash_waiting = asyncio.Event()

        async def input_ready(self) -> str:
            self.hash_waiting.set()
            await self.hash_release.wait()
            self._input_hash_ready = True
            return self.input_hash

    prepared = HashPendingPrepared()

    async def prepare(*args: object) -> FakePrepared:
        del args
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare), now=lambda: NOW
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    commit = asyncio.create_task(
        coordinator.finalize(
            draft_key, revision=2, promote_revision=1, content="question"
        )
    )
    try:
        await asyncio.wait_for(prepared.hash_waiting.wait(), timeout=0.2)
        assert not commit.done()
        assert not prepared.initial_done()
        if revoke:
            await coordinator.cancel(draft_key, revision=2)
            with pytest.raises(VoiceDraftRevisionConflict):
                await asyncio.wait_for(commit, timeout=0.2)
        else:
            prepared.hash_release.set()
            lease = await asyncio.wait_for(commit, timeout=0.2)
            assert lease is not None and lease.prepared is prepared
            assert not prepared.initial_done()
            await lease.prepared.cancel()
    finally:
        commit.cancel()
        await asyncio.gather(commit, return_exceptions=True)
        await coordinator.close()


@pytest.mark.asyncio
async def test_draft_events_continue_until_all_preparation_is_ready() -> None:
    class ContinuingPrepared(FakePrepared):
        def __init__(self) -> None:
            super().__init__(1)
            self.ready.set()
            self.continuation_ready = asyncio.Event()

        def preparation_done(self) -> bool:
            return self.continuation_ready.is_set()

        async def preparation_ready(self) -> None:
            await self.continuation_ready.wait()

    prepared = ContinuingPrepared()

    async def prepare(*args: object) -> FakePrepared:
        del args
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare), now=lambda: NOW
    )
    draft_key = key()
    await prepared.sink.append("initial")
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(
        lambda: coordinator._entries[draft_key].output.content == "initial"
    )
    events = coordinator.iter_events(draft_key, revision=1)
    await anext(events)
    assert not coordinator._entries[draft_key].ready
    await prepared.sink.append(" + recall continuation")
    await prepared.sink.flush()
    prepared.continuation_ready.set()
    seen = []
    try:
        while True:
            event = await asyncio.wait_for(anext(events), timeout=0.2)
            seen.append(event)
            if event.event == "draft.ready":
                break
        assert seen[-1].data["content"] == "initial + recall continuation"
        assert any(event.event == "draft.segment" for event in seen)
        assert coordinator._entries[draft_key].ready
    finally:
        await events.aclose()
        await coordinator.close()


@pytest.mark.asyncio
async def test_reply_context_is_frozen_with_the_revision_and_passed_to_prepare() -> (
    None
):
    reply_context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=401))
    different_context = AgentReplyContext(
        AgentMessageSubjectKind.RUN_RESULT, UUID(int=402)
    )
    prepared = FakePrepared(1)
    received: list[tuple[datetime, AgentReplyContext | None]] = []

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
        *,
        created_at: datetime,
        reply_context: AgentReplyContext | None,
    ) -> FakePrepared:
        del draft_key, revision, content
        received.append((created_at, reply_context))
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare), now=lambda: NOW
    )
    draft_key = key()
    try:
        await coordinator.upsert(
            draft_key, revision=1, content="reply", reply_context=reply_context
        )
        await wait_until(lambda: bool(received))
        assert received == [(NOW, reply_context)]
        with pytest.raises(VoiceDraftRevisionConflict):
            await coordinator.upsert(
                draft_key, revision=1, content="reply", reply_context=different_context
            )
        await coordinator.cancel(draft_key, revision=1)
        with pytest.raises(VoiceDraftRevisionConflict):
            await coordinator.upsert(
                draft_key, revision=1, content="reply", reply_context=different_context
            )
    finally:
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_context", [False, True])
async def test_finalize_only_promotes_the_frozen_reply_context(
    same_context: bool,
) -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer), now=lambda: NOW
    )
    draft_key = key()
    reply_context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=401))
    committed_context = (
        reply_context
        if same_context
        else AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=402))
    )
    try:
        await coordinator.upsert(
            draft_key, revision=1, content="reply", reply_context=reply_context
        )
        await wait_until(lambda: len(preparer.items) == 1)
        lease = await coordinator.finalize(
            draft_key,
            revision=2,
            promote_revision=1,
            content="reply",
            context_fingerprint=reply_context_fingerprint(committed_context),
        )
        if same_context:
            assert lease is not None and lease.prepared is preparer.items[0]
            assert not preparer.items[0].cancelled
            await lease.prepared.cancel()
        else:
            assert lease is None
            await wait_until(lambda: preparer.items[0].cancelled)
        assert len(preparer.items) == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_absent_entry_finalize_fences_delayed_put_at_every_revision() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    put_release = asyncio.Event()

    async def delayed_put() -> VoiceDraftSnapshot | None:
        await put_release.wait()
        return await coordinator.upsert(
            draft_key,
            revision=1,
            content="late draft",
        )

    put_task = asyncio.create_task(delayed_put())
    lease = await coordinator.finalize(
        draft_key,
        revision=4,
        promote_revision=None,
        content="authoritative",
    )
    put_release.set()
    late = await put_task
    equal = await coordinator.upsert(
        draft_key,
        revision=4,
        content="authoritative",
    )
    higher = await coordinator.upsert(
        draft_key,
        revision=99,
        content="much later",
    )
    cancelled = await coordinator.cancel(draft_key, revision=100)

    assert lease is None
    assert late is not None and late.state is VoiceDraftState.COMMITTED
    assert equal is not None and equal.state is VoiceDraftState.COMMITTED
    assert higher is not None and higher.state is VoiceDraftState.COMMITTED
    assert cancelled is not None and cancelled.state is VoiceDraftState.COMMITTED
    assert coordinator._tombstones[draft_key].revision == 4
    assert coordinator._tombstones[draft_key].content == "authoritative"
    assert preparer.items == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_authoritative_finalize_overrides_higher_cancel_tombstone() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()

    await coordinator.upsert(draft_key, revision=9, content="canceled")
    await coordinator.cancel(draft_key, revision=9)
    lease = await coordinator.finalize(
        draft_key,
        revision=4,
        promote_revision=None,
        content="authoritative",
    )
    late = await coordinator.upsert(
        draft_key,
        revision=99,
        content="late",
    )

    assert lease is None
    assert late is not None
    assert late.state is VoiceDraftState.COMMITTED
    assert late.revision == 4
    assert late.content == "authoritative"
    assert draft_key not in coordinator._entries
    await coordinator.close()


@pytest.mark.asyncio
async def test_finalize_notifies_draft_revision_but_records_commit_high_water() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, preparer),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="authoritative")
    await wait_until(lambda: len(preparer.items) == 1)
    events = coordinator.iter_events(draft_key, revision=1)
    await anext(events)

    lease = await coordinator.finalize(
        draft_key,
        revision=3,
        promote_revision=1,
        content="authoritative",
    )
    terminal = await asyncio.wait_for(anext(events), timeout=0.2)
    tombstone = coordinator._tombstones[draft_key]

    assert lease is not None
    assert lease.revision == 1
    assert terminal.data["revision"] == 1
    assert terminal.data["state"] == "committed"
    assert tombstone.revision == 3
    assert tombstone.content == "authoritative"
    assert tombstone.candidate_hash == exact_content_hash("authoritative")
    await lease.prepared.cancel()
    await coordinator.close()


@pytest.mark.asyncio
async def test_same_revision_hash_miss_revokes_speculative_owner() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="draft question")
    await wait_until(lambda: len(preparer.items) == 1)

    lease = await coordinator.claim(
        draft_key,
        promote_revision=1,
        content_hash=exact_content_hash("authoritative question"),
    )

    assert lease is None
    await wait_until(lambda: preparer.items[0].cancelled)
    await coordinator.close()


@pytest.mark.asyncio
async def test_revision_miss_claim_revokes_newer_active_owner() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=3, content="newer")
    await wait_until(lambda: len(preparer.items) == 1)

    lease = await coordinator.claim(
        draft_key,
        promote_revision=2,
        content_hash=exact_content_hash("older"),
    )

    assert lease is None
    await wait_until(lambda: preparer.items[0].cancelled)
    assert (
        await coordinator.claim(
            draft_key,
            promote_revision=3,
            content_hash=exact_content_hash("newer"),
        )
        is None
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_high_water_fences_reused_revision() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    accepted = await coordinator.upsert(draft_key, revision=3, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    cancelled = await coordinator.cancel(
        draft_key,
        revision=3,
        state=VoiceDraftState.CANCELLED,
    )

    lower = await coordinator.upsert(draft_key, revision=2, content="older")
    duplicate = await coordinator.upsert(draft_key, revision=3, content="question")
    with pytest.raises(VoiceDraftRevisionConflict):
        await coordinator.upsert(draft_key, revision=3, content="different")

    assert accepted is not None
    assert cancelled is not None
    assert lower == cancelled
    assert duplicate == cancelled
    assert duplicate.created_at == accepted.created_at
    assert len(preparer.items) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_claim_and_expiry_keep_revision_high_water() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=0.02,
        max_concurrency=2,
        now=lambda: NOW,
    )
    claimed_key = key()
    expired_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000152"),
        turn_id=UUID("00000000-0000-0000-0000-000000000153"),
    )
    await coordinator.upsert(claimed_key, revision=4, content="claimed")
    await coordinator.upsert(expired_key, revision=4, content="expired")
    await wait_until(lambda: len(preparer.items) == 2)
    lease = await coordinator.claim(
        claimed_key,
        promote_revision=4,
        content_hash=exact_content_hash("claimed"),
    )
    await asyncio.sleep(0.04)

    claimed_lower = await coordinator.upsert(
        claimed_key,
        revision=3,
        content="old claimed",
    )
    expired_lower = await coordinator.upsert(
        expired_key,
        revision=3,
        content="old expired",
    )

    assert lease is not None
    assert claimed_lower is not None
    assert claimed_lower.state is VoiceDraftState.COMMITTED
    assert expired_lower is not None
    assert expired_lower.state is VoiceDraftState.EXPIRED
    assert len(preparer.items) == 2
    await coordinator.close()


@pytest.mark.asyncio
async def test_committed_tombstone_cannot_be_cancelled_or_resurrected() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    lease = None
    try:
        await coordinator.upsert(draft_key, revision=1, content="committed")
        await wait_until(lambda: len(preparer.items) == 1)
        preparer.items[0].ready.set()
        await wait_until(lambda: coordinator._entries[draft_key].prepared is not None)
        lease = await coordinator.claim(
            draft_key,
            promote_revision=1,
            content_hash=exact_content_hash("committed"),
        )
        assert lease is not None
        committed = coordinator._tombstones[draft_key]

        with pytest.raises(VoiceDraftRevisionConflict):
            await coordinator.upsert(
                draft_key,
                revision=1,
                content="same revision conflict",
            )
        after_cancel = await coordinator.cancel(
            draft_key,
            revision=2,
            state=VoiceDraftState.CANCELLED,
        )
        after_upsert = await coordinator.upsert(
            draft_key,
            revision=3,
            content="must not resurrect",
        )

        assert committed.state is VoiceDraftState.COMMITTED
        assert after_cancel == committed
        assert after_upsert == committed
        assert coordinator._tombstones[draft_key] == committed
        assert draft_key not in coordinator._entries
        assert len(preparer.items) == 1
    finally:
        if lease is not None:
            await lease.prepared.cancel()
        await coordinator.close()


@pytest.mark.asyncio
async def test_capacity_miss_does_not_erase_terminal_high_water() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=1,
        now=lambda: NOW,
    )
    fenced_key = key()
    capacity_key = key(
        session_id=UUID("00000000-0000-0000-0000-000000000162"),
        turn_id=UUID("00000000-0000-0000-0000-000000000163"),
    )
    await coordinator.upsert(fenced_key, revision=3, content="fenced")
    await wait_until(lambda: len(preparer.items) == 1)
    tombstone = await coordinator.cancel(
        fenced_key,
        revision=3,
        state=VoiceDraftState.CANCELLED,
    )
    await coordinator.upsert(capacity_key, revision=1, content="capacity")

    higher_miss = await coordinator.upsert(
        fenced_key,
        revision=4,
        content="higher",
    )
    lower = await coordinator.upsert(
        fenced_key,
        revision=2,
        content="lower",
    )

    assert tombstone is not None
    assert higher_miss is None
    assert lower == tombstone
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_preparation_keeps_revision_high_water() -> None:
    release = asyncio.Event()

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del draft_key, revision, content
        await release.wait()
        raise RuntimeError("preparation failed")

    coordinator = VoiceDraftCoordinator(
        prepare=prepare,  # type: ignore[arg-type]
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=3, content="failed")
    events = coordinator.iter_events(draft_key, revision=3)
    await anext(events)
    release.set()

    terminal = await asyncio.wait_for(anext(events), timeout=0.2)
    lower = await coordinator.upsert(draft_key, revision=2, content="older")
    with pytest.raises(VoiceDraftRevisionConflict):
        await coordinator.upsert(draft_key, revision=3, content="different")

    assert terminal.event == "draft.failed"
    assert lower is not None
    assert lower.revision == 3
    assert lower.error == "RuntimeError"
    await coordinator.close()


@pytest.mark.asyncio
async def test_accepted_timestamp_is_passed_to_prepare_and_lease() -> None:
    calls: list[tuple[VoiceDraftKey, int, str, datetime]] = []
    prepared = FakePrepared(1)

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
        *,
        created_at: datetime,
    ) -> FakePrepared:
        calls.append((draft_key, revision, content, created_at))
        return prepared

    coordinator = VoiceDraftCoordinator(
        prepare=prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: bool(calls))
    lease = await coordinator.claim(
        draft_key,
        promote_revision=1,
        content_hash=exact_content_hash("question"),
    )

    assert calls == [(draft_key, 1, "question", NOW)]
    assert lease is not None
    assert lease.created_at == NOW
    assert lease.created_at.tzinfo is not None
    await coordinator.close()


@pytest.mark.asyncio
async def test_naive_server_timestamp_is_rejected_before_starting_work() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: datetime(2026, 8, 30, 1, 30),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        await coordinator.upsert(key(), revision=1, content="question")

    assert preparer.items == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_expiry_cancels_owned_work_and_terminates_events() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=0.02,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    events = coordinator.iter_events(draft_key, revision=1)
    first = await anext(events)

    terminal = await asyncio.wait_for(anext(events), timeout=0.2)

    assert first.event == "draft.snapshot"
    assert terminal.event == "draft.canceled"
    assert terminal.data["state"] == VoiceDraftState.EXPIRED.value
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    assert preparer.items[0].cancelled
    assert (
        await coordinator.claim(
            draft_key,
            promote_revision=1,
            content_hash=exact_content_hash("question"),
        )
        is None
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_events_emit_snapshot_content_flush_and_revision_terminal() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    await wait_until(lambda: len(preparer.items) == 1)
    events = coordinator.iter_events(draft_key, revision=1)

    first = await anext(events)
    await preparer.items[0].sink.append("回答。")
    content = await asyncio.wait_for(anext(events), timeout=0.2)
    await preparer.items[0].sink.flush()
    segment = await asyncio.wait_for(anext(events), timeout=0.2)
    await coordinator.upsert(draft_key, revision=2, content="new question")
    terminal = await asyncio.wait_for(anext(events), timeout=0.2)

    assert first.event == "draft.snapshot"
    assert first.data["revision"] == 1
    assert content.event == "draft.snapshot"
    assert content.data["content"] == "回答。"
    assert segment.event == "draft.segment"
    assert segment.data["flushOffset"] == len("回答。")
    assert terminal.event == "draft.canceled"
    assert terminal.data["revision"] == 1
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    await coordinator.close()


@pytest.mark.asyncio
async def test_claim_terminates_revision_events() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    events = coordinator.iter_events(draft_key, revision=1)
    first = await anext(events)
    await wait_until(lambda: len(preparer.items) == 1)
    await asyncio.sleep(0)

    lease = await coordinator.claim(
        draft_key,
        promote_revision=1,
        content_hash=exact_content_hash("question"),
    )
    terminal = await asyncio.wait_for(anext(events), timeout=0.2)

    assert lease is not None
    assert first.event == "draft.snapshot"
    assert terminal.event == "draft.snapshot"
    assert terminal.data["state"] == VoiceDraftState.COMMITTED.value
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    await lease.prepared.cancel()
    await coordinator.close()


@pytest.mark.asyncio
async def test_close_terminates_revision_events() -> None:
    coordinator = VoiceDraftCoordinator(
        prepare=ImmediatePreparer(),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="question")
    events = coordinator.iter_events(draft_key, revision=1)
    await anext(events)

    await coordinator.close()
    terminal = await asyncio.wait_for(anext(events), timeout=0.2)

    assert terminal.event == "draft.canceled"
    assert terminal.data["state"] == VoiceDraftState.CANCELLED.value
    with pytest.raises(StopAsyncIteration):
        await anext(events)


@pytest.mark.asyncio
async def test_stale_cancel_cannot_revoke_newer_active_revision() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    current = await coordinator.upsert(draft_key, revision=2, content="question")
    stale = await coordinator.cancel(
        draft_key,
        revision=1,
        state=VoiceDraftState.CANCELLED,
    )
    cancelled = await coordinator.cancel(
        draft_key,
        revision=2,
        state=VoiceDraftState.CANCELLED,
    )

    assert current is not None
    assert stale is None
    assert cancelled is not None
    assert cancelled.state is VoiceDraftState.CANCELLED
    await coordinator.close()


@pytest.mark.asyncio
async def test_higher_cancel_revision_revokes_active_work_and_fences_it() -> None:
    class CancellationAwarePreparer:
        def __init__(self) -> None:
            self.started: list[asyncio.Event] = []
            self.cancelled: list[asyncio.Event] = []

        async def __call__(
            self,
            draft_key: VoiceDraftKey,
            revision: int,
            content: str,
        ) -> FakePrepared:
            del draft_key, content
            started = asyncio.Event()
            cancelled = asyncio.Event()
            self.started.append(started)
            self.cancelled.append(cancelled)
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return FakePrepared(revision)

    preparer = CancellationAwarePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()
    await coordinator.upsert(draft_key, revision=1, content="candidate one")
    await wait_until(lambda: len(preparer.started) == 1)

    cancelled = await coordinator.cancel(
        draft_key,
        revision=2,
        state=VoiceDraftState.CANCELLED,
    )
    await asyncio.wait_for(preparer.cancelled[0].wait(), timeout=0.2)
    stale = await coordinator.upsert(
        draft_key,
        revision=1,
        content="candidate one",
    )
    restarted = await coordinator.upsert(
        draft_key,
        revision=3,
        content="candidate three",
    )
    await wait_until(lambda: len(preparer.started) == 2)

    assert cancelled is not None
    assert cancelled.revision == 2
    assert cancelled.state is VoiceDraftState.CANCELLED
    assert stale == cancelled
    assert restarted is not None
    assert restarted.revision == 3
    await coordinator.close()


@pytest.mark.asyncio
async def test_cancel_before_upsert_creates_bounded_revision_fence() -> None:
    preparer = ImmediatePreparer()
    coordinator = VoiceDraftCoordinator(
        prepare=preparer,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    draft_key = key()

    assert (
        await coordinator.cancel(
            draft_key,
            revision=2,
            state=VoiceDraftState.CANCELLED,
        )
        is None
    )
    with pytest.raises(VoiceDraftRevisionConflict):
        await coordinator.upsert(draft_key, revision=1, content="late candidate")
    with pytest.raises(VoiceDraftRevisionConflict):
        await coordinator.upsert(draft_key, revision=2, content="same revision")
    restarted = await coordinator.upsert(
        draft_key,
        revision=3,
        content="new candidate",
    )
    await wait_until(lambda: len(preparer.items) == 1)

    assert restarted is not None
    assert restarted.revision == 3
    assert preparer.items[0].revision == 3
    await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("activity_context", ["", "当前活动：主持直播"])
async def test_production_preparer_uses_draft_timestamp_and_private_sink(
    activity_context: str,
) -> None:
    loader_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []
    prepared = FakePrepared(7)
    agent_target = cast(AgentResolvedTarget, object())
    task_lane_context = TaskLaneContext.empty()
    draft_messages: list[ChatMessage] = [{"role": "user", "content": "draft-message"}]

    class Loader:
        async def load_for_draft(
            self,
            fish_id: UUID,
            session_id: UUID,
            *,
            turn_id: UUID,
            content: str,
            created_at: datetime,
            tracker: object,
        ) -> LoadedAnswerContext:
            loader_calls.append(
                {
                    "fish_id": fish_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "content": content,
                    "created_at": created_at,
                    "tracker": tracker,
                }
            )
            return LoadedAnswerContext(
                messages=draft_messages,
                recall_conversation=(),
                agent_target=agent_target,
                task_lane_context=task_lane_context,
                activity_context=activity_context,
            )

    class Generation:
        def prepare_initial(
            self,
            fish_id: UUID,
            messages: object,
            recall_conversation: object,
            sink: object,
            tracker: object,
            *,
            enable_home: bool,
            agent_target: object,
            task_lane_context: object,
            prepared_reply: object,
            activity_context: str = "",
        ) -> FakePrepared:
            generation_calls.append(
                {
                    "fish_id": fish_id,
                    "messages": messages,
                    "recall_conversation": recall_conversation,
                    "sink": sink,
                    "tracker": tracker,
                    "enable_home": enable_home,
                    "agent_target": agent_target,
                    "task_lane_context": task_lane_context,
                    "prepared_reply": prepared_reply,
                    "activity_context": activity_context,
                }
            )
            return prepared

    prepare = _voice_draft_preparer(  # type: ignore[arg-type]
        Loader(),
        Generation(),
        enable_home=True,
    )
    result = await prepare(
        key(),
        7,
        "打开客厅灯",
        created_at=NOW,
    )

    assert result is prepared
    assert loader_calls[0]["created_at"] == NOW
    assert loader_calls[0]["content"] == "打开客厅灯"
    assert loader_calls[0]["turn_id"] == key().turn_id
    assert generation_calls[0]["messages"] is draft_messages
    assert isinstance(generation_calls[0]["sink"], PromotableAnswerSink)
    assert generation_calls[0]["enable_home"] is True
    assert generation_calls[0]["agent_target"] is agent_target
    assert generation_calls[0]["task_lane_context"] is task_lane_context
    assert generation_calls[0]["prepared_reply"] is None
    assert generation_calls[0]["activity_context"] == activity_context
    assert generation_calls[0]["tracker"] is loader_calls[0]["tracker"]


@pytest.mark.asyncio
async def test_handled_reply_preparation_skips_answer_context_and_lanes() -> None:
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=401))
    reply = PreparedAgentReply(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        content="继续执行",
        snapshot=AgentReplySubjectSnapshot(
            context=context,
            source_message_id=UUID(int=402),
            source_content="任务完成",
            interaction_kind=None,
            allowed_decisions=(),
            recovered_without_details=False,
            replay=None,
        ),
        disposition=AgentReplyDisposition.RESULT,
        result_action=RunResultReplyAction.CONTINUE,
        message_id=TURN_ID,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class ForbiddenLoader:
        async def load_for_draft(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("handled reply must not reload answer context")

    class ForbiddenGeneration:
        def prepare_initial(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("handled reply must not launch answer lanes")

    class Replies:
        async def prepare_reply(
            self, *args: object, **kwargs: object
        ) -> PreparedAgentReply:
            calls.append((args, kwargs))
            return reply

    prepare = _voice_draft_preparer(
        ForbiddenLoader(),  # ty: ignore[invalid-argument-type]
        ForbiddenGeneration(),  # ty: ignore[invalid-argument-type]
        enable_home=True,
        reply_service=Replies(),  # ty: ignore[invalid-argument-type]
    )
    prepared = await prepare(
        key(), 1, "继续执行", created_at=NOW, reply_context=context
    )
    try:
        assert calls == [
            ((FISH_ID, SESSION_ID, "继续执行", context), {"message_id": TURN_ID})
        ]
        assert prepared.prepared_reply is reply
        assert prepared.preparation_done()
        assert (await prepared.preparation_ready()).name == "reply"
        assert prepared.sink.snapshot().content == ""
    finally:
        await prepared.cancel()


@pytest.mark.asyncio
async def test_close_drains_late_worker_cleanup() -> None:
    prepare_release = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    items: list[FakePrepared] = []

    class LateCleanupPrepared(FakePrepared):
        def cancel(self) -> asyncio.Task[None]:
            self.cancelled = True

            async def settle() -> None:
                cleanup_started.set()
                await cleanup_release.wait()

            return asyncio.create_task(settle())

    async def prepare(
        draft_key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> FakePrepared:
        del draft_key, content
        try:
            await prepare_release.wait()
        except asyncio.CancelledError:
            await prepare_release.wait()
        item = LateCleanupPrepared(revision)
        items.append(item)
        return item

    coordinator = VoiceDraftCoordinator(
        prepare=prepare,  # type: ignore[arg-type]
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: NOW,
    )
    await coordinator.upsert(key(), revision=1, content="question")
    await asyncio.sleep(0)

    close_task = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    prepare_release.set()
    await wait_until(lambda: bool(items) and items[0].cancelled)
    await cleanup_started.wait()
    for _ in range(3):
        await asyncio.sleep(0)

    assert not close_task.done()
    cleanup_release.set()
    await asyncio.wait_for(close_task, timeout=0.5)
    await asyncio.sleep(0)
    current = asyncio.current_task()
    leaked = [
        task
        for task in asyncio.all_tasks()
        if task is not current
        and not task.done()
        and task.get_name().startswith("voice-draft-")
    ]
    assert leaked == []
