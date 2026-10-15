import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from fish_memory.agent.models import AgentResolvedTarget, AgentTaskCandidate
from fish_memory.agent.routing import (
    LocalConnectorCapabilityPolicy,
    TaskLaneContext,
)
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.answer.context import (
    DEFAULT_RECENT_CONTEXT_LIMIT,
    ChatMessage,
    add_perception_context,
    add_perception_policy,
    build_chat_messages,
)
from fish_memory.answer.environment_context import (
    EnvironmentAnswerContext,
    EnvironmentAnswerContextLoader,
    current_perception_context,
)
from fish_memory.answer.history_messages import history_role_messages
from fish_memory.answer.history_tokens import estimate_history_message_tokens
from fish_memory.answer.snapshot import (
    render_dynamic_snapshot,
    render_static_profile,
)
from fish_memory.clock import format_prompt_timestamp
from fish_memory.history.models import render_history
from fish_memory.history.service import SessionHistoryService
from fish_memory.models import Message, MessageStatus, Role, visible_message_content
from fish_memory.observations.context import ObservationContextLoader
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
)
from fish_memory.soul import SoulProfile

logger = logging.getLogger(__name__)

_RECALL_CONVERSATION_MESSAGE_LIMIT = 8
_PROACTIVE_HISTORY_MESSAGE_LIMIT = 16
_DRAFT_HISTORY_BEFORE_SEQUENCE = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class LoadedAnswerContext:
    messages: list[ChatMessage]
    recall_conversation: tuple[ChatMessage, ...]
    agent_target: AgentResolvedTarget | None = None
    task_lane_context: TaskLaneContext = field(default_factory=TaskLaneContext.empty)
    activity_context: str = ""
    speech_tags: bool = False


class HistoryStore(Protocol):
    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]: ...

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
        appended_history_token_count: int | None = None,
    ) -> list[Message]: ...


class SnapshotStore(Protocol):
    async def load_normalized_dynamic_snapshot(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> NormalizedDynamicSnapshot | None: ...

    async def load_normalized_static_profile(
        self,
        fish_id: UUID,
    ) -> NormalizedStaticProfile | None: ...


class FishSoulStore(Protocol):
    async def get_profile(self, fish_id: UUID) -> SoulProfile | None: ...


class AgentTargetStore(Protocol):
    async def resolve_for_fish(
        self,
        fish_id: UUID,
        default_target_id: UUID | None,
    ) -> AgentResolvedTarget | None: ...


class AgentTaskCandidateStore(Protocol):
    async def list_task_candidates(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        recent_terminal_limit: int,
        limit: int,
        supported_connector_types: Collection[str] | None = None,
    ) -> Sequence[AgentTaskCandidate]: ...

    async def has_recent_agent_reply(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> bool: ...


class AnswerContextLoader:
    def __init__(
        self,
        history_store: HistoryStore,
        *,
        snapshot_store: SnapshotStore | None = None,
        fish_store: FishSoulStore | None = None,
        agent_target_store: AgentTargetStore | None = None,
        agent_task_store: AgentTaskCandidateStore | None = None,
        agent_default_target_id: UUID | None = None,
        agent_capability_policy: LocalConnectorCapabilityPolicy | None = None,
        agent_runtime_state: AgentRuntimeState | None = None,
        observation_context_loader: ObservationContextLoader | None = None,
        environment_context_loader: EnvironmentAnswerContextLoader | None = None,
        session_history: SessionHistoryService | None = None,
        history_message_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
        dynamic_snapshot_max_age: timedelta = timedelta(days=7),
        now: Callable[[], datetime],
        system_prompt: str,
        system_prompt_extension: Callable[[str], str] | None = None,
        activity_context_provider: Callable[[UUID, UUID], str] | None = None,
        speech_tags_provider: Callable[[UUID], Awaitable[bool]] | None = None,
    ) -> None:
        self._history_store = history_store
        self._snapshot_store = snapshot_store
        self._fish_store = fish_store
        self._agent_target_store = agent_target_store
        self._agent_task_store = agent_task_store
        self._agent_default_target_id = agent_default_target_id
        self._agent_runtime_state = agent_runtime_state
        self._agent_capability_policy = (
            agent_capability_policy or LocalConnectorCapabilityPolicy()
        )
        self._observation_context_loader = observation_context_loader
        self._environment_context_loader = environment_context_loader
        self._session_history = session_history
        self._history_message_limit = min(
            history_message_limit, DEFAULT_RECENT_CONTEXT_LIMIT
        )
        self._dynamic_snapshot_max_age = dynamic_snapshot_max_age
        self._now = now
        self._system_prompt = system_prompt
        self._system_prompt_extension = system_prompt_extension
        self._activity_context_provider = activity_context_provider
        self._speech_tags_provider = speech_tags_provider

    async def current_speech_tags(self, fish_id: UUID) -> bool:
        return (
            await self._speech_tags_provider(fish_id)
            if self._speech_tags_provider is not None
            else False
        )

    def current_activity(self, fish_id: UUID, session_id: UUID) -> str:
        if self._activity_context_provider is None:
            return ""
        return self._activity_context_provider(fish_id, session_id)

    async def load(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        through_sequence: int,
        tracker: MessagePerformanceTracker,
    ) -> list[ChatMessage]:
        return (
            await self.load_for_generation(
                fish_id,
                session_id,
                through_sequence=through_sequence,
                tracker=tracker,
            )
        ).messages

    async def load_for_generation(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        through_sequence: int,
        tracker: MessagePerformanceTracker,
    ) -> LoadedAnswerContext:
        (
            history,
            static_profile,
            dynamic_snapshot,
            system_prompt,
            agent_target,
            agent_candidates,
            has_recent_agent_reply,
        ) = await self._load_context_sources(
            fish_id,
            session_id,
            tracker,
            agent_before_sequence=through_sequence,
            history_loader=lambda: self._load_history(
                fish_id,
                session_id,
                through_sequence,
                tracker,
            ),
        )
        current_user = max(
            (
                message
                for message in history
                if message.sequence <= through_sequence and message.role is Role.USER
            ),
            key=lambda item: item.sequence,
            default=None,
        )
        current_user_content = (
            visible_message_content(current_user) if current_user is not None else ""
        )
        observation_context, environment_context = await asyncio.gather(
            self._load_observation_context(
                fish_id,
                current_user_content,
                tracker,
            ),
            self._load_environment_context(
                fish_id,
                session_id,
                content=current_user_content,
                turn_id=current_user.id if current_user is not None else None,
                tracker=tracker,
            ),
        )
        return await self._build_loaded_context(
            history,
            fish_id=fish_id,
            session_id=session_id,
            static_profile=static_profile,
            dynamic_snapshot=dynamic_snapshot,
            system_prompt=system_prompt,
            observation_context=observation_context,
            environment_context=environment_context,
            current_user_content=current_user_content,
            activity_context=self.current_activity(fish_id, session_id),
            agent_target=agent_target,
            agent_candidates=agent_candidates,
            has_recent_agent_reply=has_recent_agent_reply,
            tracker=tracker,
        )

    async def load_for_proactive(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        tracker: MessagePerformanceTracker,
        include_image: bool = False,
    ) -> LoadedAnswerContext:
        """Read the same context sources without fabricating a user turn."""
        (
            history,
            static_profile,
            dynamic_snapshot,
            system_prompt,
            agent_target,
            agent_candidates,
            has_recent_agent_reply,
        ) = await self._load_context_sources(
            fish_id,
            session_id,
            tracker,
            agent_before_sequence=None,
            history_loader=lambda: self._load_history_readonly(
                fish_id,
                session_id,
                tracker,
                appended_history_token_count=0,
                limit=_PROACTIVE_HISTORY_MESSAGE_LIMIT,
            ),
        )
        environment_context = EnvironmentAnswerContext(False, "", None)
        if self._environment_context_loader is not None:
            environment_context = await self._environment_context_loader.load_cached(
                fish_id, include_image=include_image, include_summary=False
            )
        return await self._build_loaded_context(
            history,
            fish_id=fish_id,
            session_id=session_id,
            static_profile=static_profile,
            dynamic_snapshot=dynamic_snapshot,
            system_prompt=system_prompt,
            observation_context="",
            environment_context=environment_context,
            current_user_content="",
            activity_context=self.current_activity(fish_id, session_id),
            agent_target=agent_target,
            agent_candidates=agent_candidates,
            has_recent_agent_reply=has_recent_agent_reply,
            tracker=tracker,
            preserve_order=True,
        )

    async def load_for_draft(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        content: str,
        created_at: datetime,
        tracker: MessagePerformanceTracker,
        turn_id: UUID | None = None,
        environment_revision: int | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> LoadedAnswerContext:
        (
            history,
            static_profile,
            dynamic_snapshot,
            system_prompt,
            agent_target,
            agent_candidates,
            has_recent_agent_reply,
        ) = await self._load_context_sources(
            fish_id,
            session_id,
            tracker,
            agent_before_sequence=None,
            history_loader=lambda: self._load_history_readonly(
                fish_id,
                session_id,
                tracker,
                appended_history_token_count=estimate_history_message_tokens(content),
            ),
        )
        history = [
            *history,
            Message(
                id=uuid4(),
                fish_id=fish_id,
                session_id=session_id,
                sequence=max((message.sequence for message in history), default=0) + 1,
                role=Role.USER,
                content=content,
                status=MessageStatus.INITIALIZED,
                created_at=created_at,
            ),
        ]
        observation_context, environment_context = await asyncio.gather(
            self._load_observation_context(
                fish_id,
                content,
                tracker,
            ),
            self._load_environment_context(
                fish_id,
                session_id,
                content=content,
                turn_id=turn_id,
                tracker=tracker,
                environment_revision=environment_revision,
                speech_started_at=speech_started_at,
                frame_snapshot_at=frame_snapshot_at,
            ),
        )
        return await self._build_loaded_context(
            history,
            fish_id=fish_id,
            session_id=session_id,
            static_profile=static_profile,
            dynamic_snapshot=dynamic_snapshot,
            system_prompt=system_prompt,
            observation_context=observation_context,
            environment_context=environment_context,
            current_user_content=content,
            activity_context=self.current_activity(fish_id, session_id),
            agent_target=agent_target,
            agent_candidates=agent_candidates,
            has_recent_agent_reply=has_recent_agent_reply,
            tracker=tracker,
        )

    async def _load_context_sources(
        self,
        fish_id: UUID,
        session_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        agent_before_sequence: int | None,
        history_loader: Callable[[], Awaitable[list[Message]]],
    ) -> tuple[
        list[Message],
        str,
        str,
        str,
        AgentResolvedTarget | None,
        tuple[AgentTaskCandidate, ...],
        bool,
    ]:
        with tracker.measure("context.load_total"):
            context_tasks = (
                asyncio.create_task(self._await_history_loader(history_loader)),
                asyncio.create_task(self._load_static(fish_id, tracker)),
                asyncio.create_task(self._load_dynamic(fish_id, session_id, tracker)),
                asyncio.create_task(self._load_system_prompt(fish_id, tracker)),
                asyncio.create_task(
                    self._load_agent_context(
                        fish_id,
                        session_id,
                        tracker,
                        before_sequence=agent_before_sequence,
                    )
                ),
            )
            try:
                (
                    history,
                    static_profile,
                    dynamic_snapshot,
                    system_prompt,
                    agent_context,
                ) = await asyncio.gather(*context_tasks)
            finally:
                for task in context_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*context_tasks, return_exceptions=True)

        agent_target, agent_candidates, has_recent_agent_reply = agent_context
        return (
            history,
            static_profile,
            dynamic_snapshot,
            system_prompt,
            agent_target,
            agent_candidates,
            has_recent_agent_reply,
        )

    async def _await_history_loader(
        self,
        history_loader: Callable[[], Awaitable[list[Message]]],
    ) -> list[Message]:
        return await history_loader()

    async def _build_loaded_context(
        self,
        history: list[Message],
        *,
        fish_id: UUID,
        session_id: UUID,
        static_profile: str,
        dynamic_snapshot: str,
        system_prompt: str,
        observation_context: str,
        environment_context: EnvironmentAnswerContext,
        current_user_content: str,
        agent_target: AgentResolvedTarget | None,
        agent_candidates: Sequence[AgentTaskCandidate],
        has_recent_agent_reply: bool,
        tracker: MessagePerformanceTracker,
        preserve_order: bool = False,
        activity_context: str = "",
    ) -> LoadedAnswerContext:
        history_text = None
        history_messages = None
        speech_tags = await self.current_speech_tags(fish_id)
        if self._session_history is not None:
            current = next((m for m in reversed(history) if m.role is Role.USER), None)
            history_chunks = await self._session_history.load_chunks(
                fish_id,
                session_id,
                exclude=frozenset({current.id})
                if current and not preserve_order
                else frozenset(),
            )
            history_text = render_history(history_chunks)
            history_messages = history_role_messages(history_chunks)
        with tracker.measure("context.build"):
            system_prompt += activity_context
            request_time = None
            if history_text is not None:
                request_time = f"当前时间：{format_prompt_timestamp(self._now())}"
            if self._system_prompt_extension is not None and current_user_content:
                system_prompt += self._system_prompt_extension(current_user_content)
            messages = build_chat_messages(
                system_prompt,
                history,
                static_profile=static_profile,
                dynamic_snapshot=dynamic_snapshot,
                preserve_order=preserve_order,
                history_text=history_text,
            )
            if speech_tags:
                messages[0]["speech_tags"] = True
            if history_messages is not None:
                for message in messages:
                    if message.get("context_reference") and message[
                        "content"
                    ].startswith("History:\n"):
                        message["history_messages"] = history_messages
            if activity_context:
                # Route-specific builders may replace the dialogue system
                # prompt; keep the live responsibility available to them.
                messages[0]["activity_context"] = activity_context
            add_perception_policy(
                messages,
                current_perception_context(self._environment_context_loader, fish_id),
            )
            add_perception_context(
                messages,
                observation_context=observation_context,
                summary_context=environment_context.summary_context
                if history_text is None
                else "",
                image=environment_context.image,
                images=environment_context.images,
            )
            if request_time is not None:
                messages[-1]["request_time"] = request_time
            return LoadedAnswerContext(
                messages=messages,
                activity_context=activity_context,
                speech_tags=speech_tags,
                recall_conversation=_build_recall_conversation(history),
                agent_target=agent_target,
                task_lane_context=(
                    TaskLaneContext.empty(
                        self._agent_capability_policy.routing_policy_version
                    )
                    if agent_target is None
                    else TaskLaneContext.from_records(
                        agent_candidates,
                        current_content=current_user_content,
                        capability_policy=self._agent_capability_policy,
                        has_recent_agent_reply=has_recent_agent_reply,
                    )
                ),
            )

    async def _load_observation_context(
        self,
        fish_id: UUID,
        current_user_content: str,
        tracker: MessagePerformanceTracker,
    ) -> str:
        if self._observation_context_loader is None or not current_user_content:
            return ""
        started_ns = tracker.now_ns()
        error: str | None = None
        context = ""
        try:
            context = await self._observation_context_loader.load(
                fish_id,
                current_user_content,
            )
            return context
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception(
                "Failed to load Observation context for fish %s",
                fish_id,
            )
            return ""
        finally:
            tracker.record(
                "observation.context.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                output_payload={"included": bool(context)},
                error=error,
            )

    async def _load_environment_context(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        content: str,
        turn_id: UUID | None,
        tracker: MessagePerformanceTracker,
        environment_revision: int | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> EnvironmentAnswerContext:
        if self._environment_context_loader is None or not content or turn_id is None:
            return EnvironmentAnswerContext(
                enabled=False,
                summary_context="",
                image=None,
            )
        started_ns = tracker.now_ns()
        error: str | None = None
        context = EnvironmentAnswerContext(
            enabled=False,
            summary_context="",
            image=None,
        )
        try:
            if (
                environment_revision is not None
                and speech_started_at is not None
                and frame_snapshot_at is not None
            ):
                context = await self._environment_context_loader.load(
                    fish_id,
                    session_id,
                    turn_id=turn_id,
                    environment_revision=environment_revision,
                    speech_started_at=speech_started_at,
                    frame_snapshot_at=frame_snapshot_at,
                    question=content,
                    **(
                        {"include_summary": False}
                        if self._session_history is not None
                        else {}
                    ),
                )
            else:
                context = await self._environment_context_loader.load(
                    fish_id,
                    session_id,
                    turn_id=turn_id,
                    question=content,
                    **(
                        {"include_summary": False}
                        if self._session_history is not None
                        else {}
                    ),
                )
            return context
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception(
                "Failed to load environment context for fish %s",
                fish_id,
            )
            return context
        finally:
            image = context.image
            tracker.record(
                "environment.context.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                output_payload={
                    "enabled": context.enabled,
                    "summaryIncluded": bool(context.summary_context),
                    "summarySource": context.summary_source,
                    "image": (
                        None
                        if image is None
                        else {
                            "digest": image["digest"],
                            "byteCount": image["byte_count"],
                            "width": image["width"],
                            "height": image["height"],
                            "observedAt": image["observed_at"],
                        }
                    ),
                    **(
                        {
                            "images": [
                                {
                                    "digest": item["digest"],
                                    "byteCount": item["byte_count"],
                                    "width": item["width"],
                                    "height": item["height"],
                                    "observedAt": item["observed_at"],
                                }
                                for item in context.images
                            ]
                        }
                        if context.images
                        else {}
                    ),
                },
                error=error,
            )

    async def _load_agent_context(
        self,
        fish_id: UUID,
        session_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        before_sequence: int | None,
    ) -> tuple[AgentResolvedTarget | None, tuple[AgentTaskCandidate, ...], bool]:
        if self._agent_target_store is None or self._agent_task_store is None:
            return None, (), False
        runtime = self._agent_runtime_state
        if runtime is not None and not runtime.enabled:
            return None, (), False
        revision = runtime.revision if runtime is not None else None
        started_ns = tracker.now_ns()
        error: str | None = None
        try:
            target = await self._agent_target_store.resolve_for_fish(
                fish_id,
                runtime.default_target_id
                if runtime is not None
                else self._agent_default_target_id,
            )
            if target is None:
                return None, (), False
            if not self._agent_capability_policy.supports_connector(
                str(target.endpoint.connector_type)
            ):
                return None, (), False
            candidates, has_recent_agent_reply = await asyncio.gather(
                self._agent_task_store.list_task_candidates(
                    fish_id,
                    session_id,
                    recent_terminal_limit=5,
                    limit=8,
                    supported_connector_types=(
                        self._agent_capability_policy.supported_connector_types
                    ),
                ),
                self._agent_task_store.has_recent_agent_reply(
                    fish_id,
                    session_id,
                    before_sequence=before_sequence,
                    limit=5,
                ),
            )
            if runtime is not None and (
                not runtime.enabled or runtime.revision != revision
            ):
                return None, (), False
            return (
                target,
                tuple(
                    candidate
                    for candidate in candidates
                    if self._agent_capability_policy.supports_connector(
                        str(candidate.endpoint.connector_type)
                    )
                ),
                has_recent_agent_reply,
            )
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception("Failed to load Agent task context for fish %s", fish_id)
            return None, (), False
        finally:
            tracker.record(
                "task.context.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                error=error,
            )

    async def _load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        through_sequence: int,
        tracker: MessagePerformanceTracker,
    ) -> list[Message]:
        with tracker.measure("history.load"):
            return await self._history_store.load_history(
                fish_id,
                session_id,
                before_sequence=through_sequence + 1,
                limit=self._history_message_limit,
            )

    async def _load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        appended_history_token_count: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        with tracker.measure("history.load"):
            return await self._history_store.load_history_readonly(
                fish_id,
                session_id,
                before_sequence=_DRAFT_HISTORY_BEFORE_SEQUENCE,
                limit=self._history_message_limit if limit is None else limit,
                appended_history_token_count=appended_history_token_count,
            )

    async def _load_static(
        self,
        fish_id: UUID,
        tracker: MessagePerformanceTracker,
    ) -> str:
        started_ns = tracker.now_ns()
        error: str | None = None
        try:
            if self._snapshot_store is None:
                return ""
            profile = await self._snapshot_store.load_normalized_static_profile(fish_id)
            return "" if profile is None else render_static_profile(profile)
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception("Failed to load Static Profile for fish %s", fish_id)
            return ""
        finally:
            tracker.record(
                "static.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                error=error,
            )

    async def _load_dynamic(
        self,
        fish_id: UUID,
        session_id: UUID,
        tracker: MessagePerformanceTracker,
    ) -> str:
        started_ns = tracker.now_ns()
        error: str | None = None
        try:
            if self._snapshot_store is None:
                return ""
            snapshot = await self._snapshot_store.load_normalized_dynamic_snapshot(
                fish_id,
                session_id,
            )
            if snapshot is None:
                return ""
            return render_dynamic_snapshot(
                snapshot,
                self._now(),
                self._dynamic_snapshot_max_age,
            )
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception("Failed to load Dynamic Snapshot for fish %s", fish_id)
            return ""
        finally:
            tracker.record(
                "snapshot.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                error=error,
            )

    async def _load_system_prompt(
        self,
        fish_id: UUID,
        tracker: MessagePerformanceTracker,
    ) -> str:
        started_ns = tracker.now_ns()
        error: str | None = None
        try:
            if self._fish_store is None:
                return self._system_prompt
            profile = await self._fish_store.get_profile(fish_id)
            if profile is None:
                return self._system_prompt
            return profile.render(self._system_prompt)
        except asyncio.CancelledError:
            error = "CancelledError"
            raise
        except Exception as exception:
            error = type(exception).__name__
            logger.exception("Failed to load Fish soul for fish %s", fish_id)
            return self._system_prompt
        finally:
            tracker.record(
                "soul.load",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                error=error,
            )


def _build_recall_conversation(
    history: list[Message],
) -> tuple[ChatMessage, ...]:
    ordered = sorted(history, key=lambda item: item.sequence)
    current_user = next(
        (message for message in reversed(ordered) if message.role is Role.USER),
        None,
    )
    context: list[ChatMessage] = []
    for message in ordered:
        if message is current_user:
            continue
        content = visible_message_content(message)
        if not content:
            continue
        context.append(
            {
                "role": "user" if message.role is Role.USER else "assistant",
                "content": content,
            }
        )
    return tuple(context[-_RECALL_CONVERSATION_MESSAGE_LIMIT:])
