import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory import failures
from fish_memory.agent.models import AgentResolvedTarget
from fish_memory.agent.routing import (
    AgentTaskRouteCommitter,
    LocalConnectorCapabilityPolicy,
    TaskLaneContext,
)
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.context_loader import (
    AgentTargetStore,
    AgentTaskCandidateStore,
    AnswerContextLoader,
    FishSoulStore,
    HistoryStore,
    SnapshotStore,
)
from fish_memory.answer.environment_context import EnvironmentAnswerContextLoader
from fish_memory.answer.generation import (
    AnswerGenerationError,
    RecallService,
    latest_user_content,
)
from fish_memory.answer.lifecycle import (
    AnswerLifecycle,
    PerformanceStepStore,
    TerminalMessageStore,
)
from fish_memory.answer.output_sink import RuntimeAnswerSink
from fish_memory.answer.parallel_generation import (
    HomeAnswerService,
    HomeRunResultHandle,
    ParallelAnswerGenerationPipeline,
)
from fish_memory.answer.prepared_answer import PreparedAnswer
from fish_memory.answer.rquery import RQueryIntegrityService
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.answer.voice_drafts import VoiceDraftLease
from fish_memory.db.message_attachments import MessageAttachmentStore
from fish_memory.experimental.service import ExperimentalService
from fish_memory.history.service import SessionHistoryService
from fish_memory.home.models import HomeIntentType
from fish_memory.llm.client import ChatClient
from fish_memory.models import Message, MessageStatus
from fish_memory.observations.context import ObservationContextLoader
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion

logger = logging.getLogger(__name__)


class MessageStore(Protocol):
    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None: ...

    async def insert_generating_answer(
        self,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        *,
        message_id: UUID | None = None,
        consumed_offset: int = -1,
    ) -> Message: ...

    async def update_consumed_offset(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None: ...

    async def interrupt_terminal_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None: ...

    async def soft_delete_fish(self, fish_id: UUID) -> bool: ...


class CoordinatorStore(MessageStore, HistoryStore, TerminalMessageStore, Protocol):
    pass


class HomeActionService(HomeAnswerService, Protocol):
    def start(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        content: str,
        message_sequence: int,
        *,
        intent: HomeIntentType,
        origin_ns: int | None = None,
    ) -> HomeRunResultHandle: ...


class AnswerCoordinator:
    def __init__(
        self,
        store: CoordinatorStore,
        llm: ChatClient,
        *,
        performance_store: PerformanceStepStore | None = None,
        fish_store: FishSoulStore | None = None,
        snapshot_store: SnapshotStore | None = None,
        recall_service: RecallService | None = None,
        rquery_integrity_service: RQueryIntegrityService | None = None,
        recall_context_token_budget: int = 50_000,
        answer_history_message_limit: int = 50,
        dynamic_snapshot_max_age: timedelta = timedelta(days=7),
        answer_enable_search: bool = False,
        home_action_service: HomeActionService | None = None,
        agent_target_store: AgentTargetStore | None = None,
        agent_task_store: AgentTaskCandidateStore | None = None,
        agent_default_target_id: UUID | None = None,
        agent_capability_policy: LocalConnectorCapabilityPolicy | None = None,
        agent_runtime_state: AgentRuntimeState | None = None,
        observation_context_loader: ObservationContextLoader | None = None,
        environment_context_loader: EnvironmentAnswerContextLoader | None = None,
        session_history: SessionHistoryService | None = None,
        agent_task_route_service: AgentTaskRouteCommitter | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        system_prompt: str,
        system_prompt_extension: Callable[[str], str] | None = None,
        activity_context_provider: Callable[[UUID, UUID], str] | None = None,
        speech_tags_provider: Callable[[UUID], Awaitable[bool]] | None = None,
        on_ask_submitted: Callable[[UUID, UUID], None] | None = None,
        experiment: ExperimentalService | None = None,
        message_attachments: MessageAttachmentStore | None = None,
    ) -> None:
        self._store = store
        self._context_loader = AnswerContextLoader(
            store,
            snapshot_store=snapshot_store,
            fish_store=fish_store,
            agent_target_store=agent_target_store,
            agent_task_store=agent_task_store,
            agent_default_target_id=agent_default_target_id,
            agent_capability_policy=agent_capability_policy,
            agent_runtime_state=agent_runtime_state,
            observation_context_loader=observation_context_loader,
            environment_context_loader=environment_context_loader,
            session_history=session_history,
            history_message_limit=answer_history_message_limit,
            dynamic_snapshot_max_age=dynamic_snapshot_max_age,
            now=now,
            system_prompt=system_prompt,
            system_prompt_extension=system_prompt_extension,
            activity_context_provider=activity_context_provider,
            speech_tags_provider=speech_tags_provider,
        )
        self._generation = ParallelAnswerGenerationPipeline(
            llm,
            recall_service=recall_service,
            rquery_integrity_service=rquery_integrity_service,
            recall_context_token_budget=recall_context_token_budget,
            enable_search=answer_enable_search,
            task_route_service=agent_task_route_service,
            home_service=home_action_service,
            experiment=experiment,
            attachments=message_attachments,
        )
        self._home_action_service = home_action_service
        self._on_ask_submitted = on_ask_submitted
        self._lifecycle = AnswerLifecycle(
            store,
            performance_store=performance_store,
        )
        self._runtimes: dict[tuple[UUID, UUID], FishRuntime] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._background_tasks_by_fish: dict[UUID, set[asyncio.Task[None]]] = {}
        self._unacknowledged_voice_leases: dict[
            asyncio.Task[None],
            tuple[UUID, VoiceDraftLease],
        ] = {}
        self._voice_lease_cleanup_tasks: dict[asyncio.Task[None], UUID] = {}

    def get_runtime(self, fish_id: UUID, session_id: UUID) -> FishRuntime:
        key = (fish_id, session_id)
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = FishRuntime()
            self._runtimes[key] = runtime
        return runtime

    def peek_runtime(self, fish_id: UUID, session_id: UUID) -> FishRuntime | None:
        return self._runtimes.get((fish_id, session_id))

    def has_active_generation(self) -> bool:
        return any(
            runtime.llm_task is not None and not runtime.llm_task.done()
            for runtime in self._runtimes.values()
        )

    def _find_answer_runtime(
        self,
        fish_id: UUID,
        message_id: UUID,
    ) -> FishRuntime | None:
        for (runtime_fish_id, _session_id), runtime in self._runtimes.items():
            if runtime_fish_id == fish_id and runtime.answer_message_id == message_id:
                return runtime
        return None

    def answer_playback_owner(self, fish_id: UUID, message_id: UUID) -> str:
        runtime = self._find_answer_runtime(fish_id, message_id)
        return runtime.playback_owner if runtime is not None else "web"

    def notify_user_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        message_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        source_message: Message | None = None,
        playback_owner: str = "web",
    ) -> None:
        if source_message is not None and (
            source_message.id != message_id
            or source_message.fish_id != fish_id
            or source_message.session_id != session_id
            or source_message.sequence != sequence
        ):
            raise ValueError("source_message does not match the notified user message")
        queued_ns = tracker.now_ns()
        task = asyncio.create_task(
            self.handle_user_message(
                fish_id,
                session_id,
                sequence,
                message_id=message_id,
                tracker=tracker,
                queued_ns=queued_ns,
                source_message=source_message,
                playback_owner=playback_owner,
            )
        )
        self._background_tasks.add(task)
        self._background_tasks_by_fish.setdefault(fish_id, set()).add(task)
        task.add_done_callback(
            lambda completed: self._background_task_done(fish_id, completed)
        )

    def notify_voice_turn(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        message_id: UUID,
        answer_message_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        source_message: Message,
        draft_lease: VoiceDraftLease | None,
    ) -> None:
        if (
            source_message.id != message_id
            or source_message.fish_id != fish_id
            or source_message.session_id != session_id
            or source_message.sequence != sequence
        ):
            raise ValueError("source_message does not match the notified user message")
        if draft_lease is not None and draft_lease.key != VoiceDraftKey(
            fish_id,
            session_id,
            message_id,
        ):
            raise ValueError("draft_lease does not match the notified voice turn")
        queued_ns = tracker.now_ns()
        task = asyncio.create_task(
            self._handle_notified_voice_turn(
                fish_id,
                session_id,
                sequence,
                message_id=message_id,
                answer_message_id=answer_message_id,
                tracker=tracker,
                queued_ns=queued_ns,
                source_message=source_message,
                draft_lease=draft_lease,
            )
        )
        if draft_lease is not None:
            self._unacknowledged_voice_leases[task] = (fish_id, draft_lease)
        self._background_tasks.add(task)
        self._background_tasks_by_fish.setdefault(fish_id, set()).add(task)
        task.add_done_callback(
            lambda completed: self._background_task_done(fish_id, completed)
        )

    async def _handle_notified_voice_turn(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        *,
        message_id: UUID,
        answer_message_id: UUID,
        tracker: MessagePerformanceTracker,
        queued_ns: int,
        source_message: Message,
        draft_lease: VoiceDraftLease | None,
    ) -> None:
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("voice notification has no owning task")
        await self.handle_user_message(
            fish_id,
            session_id,
            sequence,
            message_id=message_id,
            answer_message_id=answer_message_id,
            tracker=tracker,
            queued_ns=queued_ns,
            source_message=source_message,
            draft_lease=draft_lease,
            voice_lease_owner=owner_task,
        )

    def _background_task_done(
        self,
        fish_id: UUID,
        task: asyncio.Task[None],
    ) -> None:
        self._discard_background_task(fish_id, task)
        ownership = self._unacknowledged_voice_leases.pop(task, None)
        if ownership is not None:
            owner_fish_id, lease = ownership
            self._schedule_voice_lease_cleanup(owner_fish_id, lease)

    def _discard_background_task(
        self,
        fish_id: UUID,
        task: asyncio.Task[None],
    ) -> None:
        self._background_tasks.discard(task)
        fish_tasks = self._background_tasks_by_fish.get(fish_id)
        if fish_tasks is None:
            return
        fish_tasks.discard(task)
        if not fish_tasks:
            self._background_tasks_by_fish.pop(fish_id, None)

    async def close(self) -> None:
        tasks = set(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._settle_unacknowledged_voice_leases()
        await self._drain_voice_lease_cleanups()
        self._background_tasks.clear()
        self._background_tasks_by_fish.clear()
        for (fish_id, _session_id), runtime in self._runtimes.items():
            async with runtime.generation_lock:
                await self._interrupt_current_answer(fish_id, runtime)
        await self._lifecycle.close()

    async def delete_fish(self, fish_id: UUID) -> bool:
        await self._cancel_background_tasks(fish_id)
        runtimes = [
            (key, runtime)
            for key, runtime in self._runtimes.items()
            if key[0] == fish_id
        ]
        for _key, runtime in runtimes:
            async with runtime.generation_lock:
                await self._cancel_generation_for_deletion(runtime)
        deleted = await self._store.soft_delete_fish(fish_id)
        for key, runtime in runtimes:
            if self._runtimes.get(key) is runtime:
                self._runtimes.pop(key)
        return deleted

    async def _cancel_background_tasks(self, fish_id: UUID) -> None:
        tasks = set(self._background_tasks_by_fish.get(fish_id, ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._settle_unacknowledged_voice_leases(fish_id)
        await self._drain_voice_lease_cleanups(fish_id)

    async def _settle_unacknowledged_voice_leases(
        self,
        fish_id: UUID | None = None,
    ) -> None:
        leases: list[VoiceDraftLease] = []
        for task, (owner_fish_id, lease) in tuple(
            self._unacknowledged_voice_leases.items()
        ):
            if fish_id is not None and owner_fish_id != fish_id:
                continue
            self._unacknowledged_voice_leases.pop(task, None)
            leases.append(lease)
        if leases:
            await asyncio.gather(
                *(self._cancel_voice_lease(lease) for lease in leases),
                return_exceptions=True,
            )

    def _schedule_voice_lease_cleanup(
        self,
        fish_id: UUID,
        lease: VoiceDraftLease,
    ) -> None:
        cleanup = asyncio.create_task(
            self._cancel_voice_lease(lease),
            name=f"voice-lease-cleanup:{lease.key.turn_id}",
        )
        self._voice_lease_cleanup_tasks[cleanup] = fish_id
        cleanup.add_done_callback(self._voice_lease_cleanup_done)

    def _voice_lease_cleanup_done(self, task: asyncio.Task[None]) -> None:
        self._voice_lease_cleanup_tasks.pop(task, None)
        with suppress(asyncio.CancelledError):
            task.exception()

    async def _drain_voice_lease_cleanups(
        self,
        fish_id: UUID | None = None,
    ) -> None:
        while True:
            cleanup = tuple(
                task
                for task, owner_fish_id in self._voice_lease_cleanup_tasks.items()
                if fish_id is None or owner_fish_id == fish_id
            )
            if not cleanup:
                return
            await asyncio.gather(*cleanup, return_exceptions=True)

    @staticmethod
    async def _cancel_voice_lease(lease: VoiceDraftLease) -> None:
        try:
            await lease.prepared.cancel()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to settle voice lease for turn %s",
                lease.key.turn_id,
            )

    async def _cancel_generation_for_deletion(self, runtime: FishRuntime) -> None:
        task = runtime.llm_task
        if task is None or task.done():
            return

        async with runtime.condition:
            runtime.status = MessageStatus.INTERRUPTED
            runtime.source_message_id = None
            runtime.performance_tracker = None
            runtime.condition.notify_all()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def update_progress(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> int:
        if consumed_offset < 0:
            raise ValueError("consumed offset must be non-negative")

        runtime = self._find_answer_runtime(fish_id, message_id)
        if runtime is not None:
            async with runtime.condition:
                if runtime.status is MessageStatus.INTERRUPTED:
                    return max(0, runtime.consumed_offset)
                if consumed_offset > len(runtime.content):
                    raise ValueError("consumed offset exceeds generated content")

        stored_offset = await self._store.update_consumed_offset(
            message_id,
            fish_id,
            consumed_offset,
        )
        if stored_offset is None:
            raise LookupError(str(message_id))

        if runtime is not None:
            async with runtime.condition:
                runtime.consumed_offset = max(
                    runtime.consumed_offset,
                    stored_offset,
                )
        return stored_offset

    async def interrupt_answer(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        """Stop the active answer and persist only consumed content."""
        if consumed_offset < 0:
            raise ValueError("consumed offset must be non-negative")

        runtime = self._find_answer_runtime(fish_id, message_id)
        if runtime is None:
            return
        async with runtime.generation_lock:
            async with runtime.condition:
                if runtime.answer_message_id != message_id:
                    return
                current_status = runtime.status
                if current_status is MessageStatus.INTERRUPTED:
                    return
                if consumed_offset > len(runtime.content):
                    raise ValueError("consumed offset exceeds generated content")
                if current_status is MessageStatus.GENERATING:
                    runtime.consumed_offset = max(
                        runtime.consumed_offset,
                        consumed_offset,
                    )
            if current_status is MessageStatus.GENERATING:
                await self._interrupt_current_answer(fish_id, runtime)
                return
            if current_status is MessageStatus.FAILED:
                # Failed generation can leave playable text. Record where its
                # playback stopped without replacing the generation failure.
                await self.update_progress(fish_id, message_id, consumed_offset)
                return
            if current_status is not MessageStatus.COMPLETED:
                return

            # Text generation often finishes before TTS playback. Treat the
            # playback cutoff as authoritative even for a completed answer.
            stored_offset = await self._store.interrupt_terminal_answer(
                message_id,
                fish_id,
                consumed_offset,
            )
            if stored_offset is None:
                return
            async with runtime.condition:
                if runtime.answer_message_id != message_id:
                    return
                runtime.content = runtime.content[:stored_offset]
                runtime.flush_offset = min(
                    runtime.flush_offset,
                    len(runtime.content),
                )
                runtime.consumed_offset = stored_offset
                runtime.status = MessageStatus.INTERRUPTED
                runtime.condition.notify_all()

    async def handle_user_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        *,
        message_id: UUID,
        answer_message_id: UUID | None = None,
        tracker: MessagePerformanceTracker,
        queued_ns: int | None = None,
        source_message: Message | None = None,
        draft_lease: VoiceDraftLease | None = None,
        voice_lease_owner: asyncio.Task[None] | None = None,
        playback_owner: str = "web",
    ) -> None:
        playback_owner = (
            "companion" if answer_message_id is not None else playback_owner
        )
        runtime = self.get_runtime(fish_id, session_id)
        if source_message is not None:
            async with runtime.condition:
                current = runtime.latest_user_message
                if current is None or source_message.sequence >= current.sequence:
                    runtime.latest_user_message = source_message
                    runtime.condition.notify_all()
        queue_started_ns = tracker.now_ns() if queued_ns is None else queued_ns
        tracker_delegated = False
        owned_draft_lease = draft_lease

        async def settle_owned_draft() -> None:
            nonlocal owned_draft_lease
            if owned_draft_lease is None:
                return
            lease = owned_draft_lease
            owned_draft_lease = None
            if voice_lease_owner is not None:
                self._unacknowledged_voice_leases.pop(voice_lease_owner, None)
            await self._cancel_voice_lease(lease)

        def transfer_owned_draft() -> None:
            nonlocal owned_draft_lease
            if owned_draft_lease is None:
                return
            if voice_lease_owner is not None:
                self._unacknowledged_voice_leases.pop(voice_lease_owner, None)
            owned_draft_lease = None

        try:
            async with runtime.generation_lock:
                tracker.record(
                    "coordinator.queue_wait",
                    started_ns=queue_started_ns,
                    finished_ns=tracker.now_ns(),
                )
                if (
                    runtime.context_through_sequence is not None
                    and runtime.context_through_sequence >= sequence
                ):
                    await settle_owned_draft()
                    if tracker is not runtime.performance_tracker:
                        self._lifecycle.submit_performance(message_id, tracker)
                    tracker_delegated = True
                    return

                with tracker.measure("answer.interrupt_previous"):
                    await self._interrupt_current_answer(fish_id, runtime)

                with tracker.measure("answer.find_latest_message"):
                    latest_sequence = await self._store.get_latest_user_sequence(
                        fish_id,
                        session_id,
                    )
                if latest_sequence is None:
                    await settle_owned_draft()
                    self._lifecycle.submit_performance(message_id, tracker)
                    tracker_delegated = True
                    return
                if latest_sequence > sequence:
                    await settle_owned_draft()
                    self._lifecycle.submit_performance(message_id, tracker)
                    tracker_delegated = True
                    return
                with tracker.measure("answer.create"):
                    answer = await self._store.insert_generating_answer(
                        fish_id,
                        session_id,
                        latest_sequence,
                        message_id=answer_message_id,
                        **(
                            {"consumed_offset": 0}
                            if playback_owner == "companion"
                            else {}
                        ),
                    )
                prepared: PreparedAnswer | None = None
                if (
                    owned_draft_lease is not None
                    and owned_draft_lease.prepared.input_context.messages
                    and owned_draft_lease.prepared.input_context.speech_tags
                    != await self._context_loader.current_speech_tags(fish_id)
                ):
                    # A provider switch can change the generated speech protocol.
                    await settle_owned_draft()
                if (
                    owned_draft_lease is not None
                    and owned_draft_lease.prepared.input_context.messages
                    and owned_draft_lease.prepared.input_context.activity_context
                    != self._context_loader.current_activity(fish_id, session_id)
                ):
                    # Only a plugin activity transition invalidates this lease;
                    # ordinary frame/history changes keep the existing fast path.
                    await settle_owned_draft()
                if owned_draft_lease is not None:
                    lease = owned_draft_lease
                    prepared = lease.prepared
                    loaded_context = prepared.input_context
                    now_ns = tracker.now_ns()
                    tracker.record(
                        "voice_draft.promoted",
                        started_ns=now_ns,
                        finished_ns=now_ns,
                        output_payload={"input_hash": lease.input_hash},
                    )
                else:
                    loaded_context = await self._context_loader.load_for_generation(
                        fish_id,
                        session_id,
                        through_sequence=latest_sequence,
                        tracker=tracker,
                    )
                    if answer_message_id is not None:
                        now_ns = tracker.now_ns()
                        tracker.record(
                            "voice_draft.fallback",
                            started_ns=now_ns,
                            finished_ns=now_ns,
                        )

                # Prepared voice turns retain the context frozen at their pause.
                # The next recorder checkpoint belongs to a later turn.
                self._notify_ask_submitted(fish_id, session_id)

                async with runtime.condition:
                    runtime.answer_message_id = answer.id
                    runtime.answer_sequence = answer.sequence
                    runtime.context_through_sequence = latest_sequence
                    runtime.content = ""
                    runtime.speech_content = ""
                    runtime.speech_tags = loaded_context.speech_tags
                    runtime.flush_offset = 0
                    runtime.consumed_offset = 0 if playback_owner == "companion" else -1
                    runtime.playback_owner = playback_owner
                    runtime.status = MessageStatus.GENERATING
                    runtime.failure = None
                    runtime.end_conversation = False
                    runtime.pet_emotion = PetEmotion.NEUTRAL
                    runtime.pet_metadata_revision = 0
                    runtime.source_message_id = message_id
                    runtime.performance_tracker = tracker
                    runtime.llm_task = asyncio.create_task(
                        self._run_generation(
                            fish_id,
                            session_id,
                            answer.id,
                            loaded_context.messages,
                            loaded_context.recall_conversation,
                            loaded_context.agent_target,
                            loaded_context.task_lane_context,
                            latest_sequence,
                            message_id,
                            tracker,
                            prepared=prepared,
                        )
                    )
                    transfer_owned_draft()
                    tracker_delegated = True
                    runtime.condition.notify_all()
        except BaseException:
            await settle_owned_draft()
            if not tracker_delegated:
                self._lifecycle.submit_performance(message_id, tracker)
            raise

    async def _keep_unsaid(self, message_id: UUID, content: str) -> None:
        """Best effort: showing the unsaid rest must never break an interruption."""
        record = getattr(self._store, "record_unsaid", None)
        if record is None or not content.strip():
            return
        try:
            await record(message_id, content)
        except Exception:
            logger.warning("Could not keep the unsaid rest of %s", message_id)

    async def _interrupt_current_answer(
        self,
        fish_id: UUID,
        runtime: FishRuntime,
    ) -> None:
        task = runtime.llm_task
        if task is None:
            return

        async with runtime.condition:
            if (
                runtime.status is not MessageStatus.GENERATING
                or runtime.answer_message_id is None
            ):
                return
            answer_message_id = runtime.answer_message_id
            source_message_id = runtime.source_message_id
            tracker = runtime.performance_tracker
            unsaid = ""
            if runtime.consumed_offset >= 0:
                unsaid = runtime.content[runtime.consumed_offset :]
                runtime.content = runtime.content[: runtime.consumed_offset]
                runtime.flush_offset = min(
                    runtime.flush_offset,
                    len(runtime.content),
                )
            runtime.status = MessageStatus.INTERRUPTED
            runtime.condition.notify_all()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # A failed task still owns its generating row until this cutoff is saved.
            logger.exception(
                "Generation task failed before interruption: fish_id=%s, message_id=%s",
                fish_id,
                answer_message_id,
            )
        await self._keep_unsaid(answer_message_id, unsaid)

        async with runtime.condition:
            if runtime.answer_message_id != answer_message_id:
                return
            content = runtime.content
            consumed_offset = runtime.consumed_offset

        if consumed_offset >= 0:
            try:
                stored_offset = await self._store.update_consumed_offset(
                    answer_message_id,
                    fish_id,
                    min(consumed_offset, len(content)),
                )
            except Exception:
                logger.exception(
                    "Failed to persist consumed offset while interrupting answer: "
                    "fish_id=%s, message_id=%s",
                    fish_id,
                    answer_message_id,
                )
            else:
                if stored_offset is not None:
                    consumed_offset = min(stored_offset, len(content))
                    content = content[:consumed_offset]
                    async with runtime.condition:
                        if runtime.answer_message_id == answer_message_id:
                            runtime.content = content
                            runtime.flush_offset = min(
                                runtime.flush_offset,
                                len(content),
                            )
                            runtime.condition.notify_all()

        if source_message_id is None or tracker is None:
            return
        await self._lifecycle.interrupt(
            fish_id,
            answer_message_id,
            source_message_id,
            runtime,
            tracker,
        )

    async def _run_generation(
        self,
        fish_id: UUID,
        session_id: UUID,
        answer_message_id: UUID,
        messages: list[ChatMessage],
        recall_conversation: tuple[ChatMessage, ...],
        agent_target: AgentResolvedTarget | None,
        task_lane_context: TaskLaneContext,
        source_sequence: int,
        source_message_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        prepared: PreparedAnswer | None = None,
    ) -> None:
        runtime = self.get_runtime(fish_id, session_id)
        home_action_service = self._home_action_service
        start_home_action = (
            None
            if home_action_service is None
            else lambda intent: home_action_service.start(
                fish_id,
                session_id,
                source_message_id,
                latest_user_content(messages),
                source_sequence,
                intent=intent,
                origin_ns=tracker.origin_ns,
            )
        )
        failed = False
        failure: failures.ServiceFailure | None = None
        try:
            if prepared is None:
                await self._generation.generate(
                    fish_id,
                    answer_message_id,
                    messages,
                    recall_conversation,
                    runtime,
                    tracker,
                    source_message_id=source_message_id,
                    agent_target=agent_target,
                    task_lane_context=task_lane_context,
                    start_home_action=start_home_action,
                    session_id=session_id,
                )
            else:
                await prepared.release_to_public(
                    RuntimeAnswerSink(runtime, answer_message_id, tracker),
                    source_message_id=source_message_id,
                    start_home_action=start_home_action,
                    answer_message_id=answer_message_id,
                    runtime=runtime,
                    session_id=session_id,
                )
                route_evidence_builder = getattr(prepared, "route_evidence", None)
                route_evidence = (
                    route_evidence_builder()
                    if callable(route_evidence_builder)
                    else None
                )
                if route_evidence is not None:
                    now_ns = tracker.now_ns()
                    tracker.record(
                        "voice_draft.route",
                        started_ns=now_ns,
                        finished_ns=now_ns,
                        output_payload=route_evidence,
                    )
        except asyncio.CancelledError:
            if prepared is not None:
                await prepared.cancel()
            raise
        except AnswerGenerationError as error:
            if prepared is not None:
                await prepared.cancel()
            failed, failure = True, failures.failure_of(error)
        except Exception as error:
            if prepared is not None:
                await prepared.cancel()
            logger.exception(
                "Answer generation failed: fish_id=%s, message_id=%s",
                fish_id,
                answer_message_id,
            )
            failed, failure = True, failures.failure_of(error)
        except BaseException:
            if prepared is not None:
                await prepared.cancel()
            raise
        finally:
            if prepared is not None:
                export = getattr(prepared, "record_preparation_performance", None)
                if callable(export):
                    try:
                        export(tracker)
                    except Exception:
                        logger.exception("Unable to record voice draft preparation")
        if failed:
            await self._lifecycle.fail(
                fish_id,
                answer_message_id,
                source_message_id,
                runtime,
                tracker,
                failure.code if failure is not None else None,
            )
            return
        await self._lifecycle.complete(
            fish_id,
            answer_message_id,
            source_message_id,
            runtime,
            tracker,
        )

    def _notify_ask_submitted(self, fish_id: UUID, session_id: UUID) -> None:
        callback = self._on_ask_submitted
        if callback is None:
            return
        try:
            callback(fish_id, session_id)
        except Exception:
            logger.exception(
                "Ask submission callback failed for fish %s session %s",
                fish_id,
                session_id,
            )
