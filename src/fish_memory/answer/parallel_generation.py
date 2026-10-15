from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from fish_memory.agent.detection import looks_like_agent_task_request
from fish_memory.agent.models import AgentResolvedTarget
from fish_memory.agent.prompt import build_task_lane_messages
from fish_memory.agent.replies import PreparedAgentReply
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteCommitter,
    PreparedAgentTaskRoute,
    TaskLaneContext,
)
from fish_memory.answer.context import (
    ChatImageAttachment,
    ChatMessage,
    ChatUserAttachment,
    build_web_search_context_messages,
    get_environment_images,
    parse_search_context,
)
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.generation import (
    AnswerGenerationError,
    AnswerGenerationPipeline,
    RecallService,
    _join_answer_append,
    _LlmStreamError,
    latest_user_content,
)
from fish_memory.answer.lane_context import (
    DEFAULT_RECENT_CONTEXT_LIMIT,
    build_home_lane_messages,
    build_lane_base_messages,
    build_recall_lane_messages,
    build_web_lane_messages,
)
from fish_memory.answer.lane_runtime import (
    AnswerLaneArbitrator,
    AnswerLaneDefinition,
    AnswerLaneHandle,
    AnswerLaneInvocation,
    AnswerLaneResult,
    AnswerLaneRunner,
)
from fish_memory.answer.lane_stream import (
    HomeLaneStreamParser,
    RecallLaneStreamParser,
    TaskLaneStreamParser,
    WebLaneStreamParser,
)
from fish_memory.answer.output_sink import (
    PromotableAnswerSink,
    RuntimeAnswerSink,
)
from fish_memory.answer.prepared_answer import (
    HomeRunResultHandle,
    PreparedAnswer,
    PreparedInitialAnswer,
)
from fish_memory.answer.rquery import RQueryIntegrityService, RQueryResolution
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.speech_style import speech_tags_enabled, with_speech_tag_rules
from fish_memory.answer.voice_draft_types import (
    exact_content_hash,
    semantic_initial_input_hash,
)
from fish_memory.db.message_attachments import (
    MessageAttachment,
    MessageAttachmentStore,
)
from fish_memory.experimental.prepared import ExperimentalPreparedAnswer
from fish_memory.experimental.service import ExperimentalService
from fish_memory.home.answer import build_home_query_continuation_messages
from fish_memory.home.detection import looks_like_home_request
from fish_memory.home.models import HomeIntentType, HomeRunOutcome
from fish_memory.home.preparation import PreparedHomeRequest
from fish_memory.llm.client import ChatClient
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion
from fish_memory.recall.models import RecallResult

logger = logging.getLogger(__name__)

_RECALL_LANE_NAME = "recall"
_WEB_LANE_NAME = "web"
_HOME_LANE_NAME = "home"
_TASK_LANE_NAME = "task"


def _build_recall_result(values: Mapping[str, object]) -> AnswerLaneResult:
    _require_emotion(values)
    depth = _require_int(values, "recall_depth", {0, 1})
    query = values.get("query")
    if depth == 1 and (not isinstance(query, str) or not query.strip()):
        raise ValueError("Recall lane depth requires a non-empty query")
    if depth == 0 and query is not None:
        raise ValueError("Recall lane depth zero must not contain a query")
    end = values.get("end")
    if not isinstance(end, bool):
        raise ValueError("Recall lane is missing its end decision")
    return AnswerLaneResult(
        name=_RECALL_LANE_NAME,
        accepted=True,
        query=query if isinstance(query, str) else None,
        recall_depth=depth,
        end_conversation=end,
    )


def _build_web_result(values: Mapping[str, object]) -> AnswerLaneResult:
    accepted = _require_bool(values, "accepted")
    if not accepted:
        return AnswerLaneResult(name=_WEB_LANE_NAME, accepted=False)
    _require_emotion(values)
    query = values.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("accepted Web lane requires a non-empty query")
    return AnswerLaneResult(
        name=_WEB_LANE_NAME,
        accepted=True,
        query=query,
    )


def _build_home_result(values: Mapping[str, object]) -> AnswerLaneResult:
    accepted = _require_bool(values, "accepted")
    if not accepted:
        return AnswerLaneResult(name=_HOME_LANE_NAME, accepted=False)
    _require_emotion(values)
    home_intent = values.get("home_intent")
    if not isinstance(home_intent, HomeIntentType):
        raise ValueError("accepted Home lane is missing its intent type")
    return AnswerLaneResult(
        name=_HOME_LANE_NAME,
        accepted=True,
        home_intent=home_intent,
    )


def _build_task_result(values: Mapping[str, object]) -> AnswerLaneResult:
    accepted = _require_bool(values, "accepted")
    if not accepted:
        return AnswerLaneResult(name=_TASK_LANE_NAME, accepted=False)
    _require_emotion(values)
    kind = values.get("task_kind")
    indexes = values.get("candidate_indexes")
    if not isinstance(kind, AgentTaskDecisionKind):
        raise ValueError("accepted Task lane is missing its decision kind")
    if not isinstance(indexes, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indexes
    ):
        raise ValueError("accepted Task lane is missing candidate indexes")
    parsed_indexes = cast(tuple[int, ...], indexes)
    return AnswerLaneResult(
        name=_TASK_LANE_NAME,
        accepted=True,
        task_kind=kind,
        candidate_indexes=parsed_indexes,
    )


def _require_emotion(values: Mapping[str, object]) -> PetEmotion:
    emotion = values.get("emotion")
    if not isinstance(emotion, PetEmotion):
        raise ValueError("accepted answer lane is missing its emotion")
    return emotion


def _require_bool(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"answer lane is missing boolean {key}")
    return value


def _require_int(
    values: Mapping[str, object],
    key: str,
    allowed: set[int],
) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value not in allowed:
        raise ValueError(f"answer lane has invalid integer {key}")
    return value


_RECALL_LANE = AnswerLaneDefinition(
    name=_RECALL_LANE_NAME,
    priority=None,
    parser_factory=RecallLaneStreamParser,
    result_builder=_build_recall_result,
    performance_step="llm.answer.initial",
    max_attempts=2,
    record_initial_timing=True,
)
_WEB_LANE = AnswerLaneDefinition(
    name=_WEB_LANE_NAME,
    priority=250,
    parser_factory=WebLaneStreamParser,
    result_builder=_build_web_result,
    performance_step="llm.answer.web_lane",
)
_HOME_LANE = AnswerLaneDefinition(
    name=_HOME_LANE_NAME,
    priority=300,
    parser_factory=HomeLaneStreamParser,
    result_builder=_build_home_result,
    performance_step="llm.answer.home_lane",
)


def _task_lane(candidate_count: int) -> AnswerLaneDefinition:
    return AnswerLaneDefinition(
        name=_TASK_LANE_NAME,
        priority=200,
        parser_factory=lambda: TaskLaneStreamParser(candidate_count=candidate_count),
        result_builder=_build_task_result,
        performance_step="llm.answer.task_lane",
    )


class HomeAnswerService(Protocol):
    async def prepare(
        self,
        content: str,
        *,
        intent: HomeIntentType,
        history: Sequence[ChatMessage],
        tracker: MessagePerformanceTracker,
    ) -> PreparedHomeRequest: ...

    def start_prepared(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        prepared: PreparedHomeRequest,
        *,
        origin_ns: int | None = None,
    ) -> HomeRunResultHandle: ...


@dataclass(slots=True)
class _RouteContext:
    fish_id: UUID
    answer_message_id: UUID | None
    recall_conversation: Sequence[ChatMessage]
    recall_base_messages: list[ChatMessage]
    web_base_messages: list[ChatMessage]
    home_base_messages: list[ChatMessage]
    runtime: FishRuntime | None
    tracker: MessagePerformanceTracker
    output_sink: PromotableAnswerSink
    original_query: str
    agent_target: AgentResolvedTarget | None
    task_lane_context: TaskLaneContext
    environment_images: tuple[ChatImageAttachment, ...]
    source_message_id: UUID | None = None
    session_id: UUID | None = None
    prepared_home: PreparedHomeRequest | None = None
    prepared_task: PreparedAgentTaskRoute | None = None


RouteHandler = Callable[[_RouteContext, AnswerLaneResult], Awaitable[None]]


class ParallelAnswerGenerationPipeline(AnswerGenerationPipeline):
    """Select one lane and prepare its full answer against a frozen input.

    Selected computation writes through a promotable sink. Home and Task
    services keep their mutation boundary separate from preparation.
    """

    def __init__(
        self,
        llm: ChatClient,
        *,
        recall_service: RecallService | None,
        rquery_integrity_service: RQueryIntegrityService | None = None,
        recall_context_token_budget: int,
        enable_search: bool,
        recent_context_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
        task_route_service: AgentTaskRouteCommitter | None = None,
        home_service: HomeAnswerService | None = None,
        experiment: ExperimentalService | None = None,
        attachments: MessageAttachmentStore | None = None,
    ) -> None:
        self._attachments = attachments
        super().__init__(
            llm,
            recall_service=recall_service,
            recall_context_token_budget=recall_context_token_budget,
            enable_search=enable_search,
        )
        if recent_context_limit < 1:
            raise ValueError("recent lane context limit must be positive")
        self._recent_context_limit = recent_context_limit
        self._lane_runner = AnswerLaneRunner(llm)
        self._arbitrator = AnswerLaneArbitrator()
        self._rquery_integrity_service = rquery_integrity_service
        self._task_route_service = task_route_service
        self._home_service = home_service
        self.experiment = experiment
        self._route_handlers: dict[str, RouteHandler] = {
            _RECALL_LANE_NAME: self._complete_recall_route,
            _WEB_LANE_NAME: self._complete_web_route,
            _HOME_LANE_NAME: self._prepare_home_route,
            _TASK_LANE_NAME: self._prepare_task_route,
        }

    async def generate(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        messages: list[ChatMessage],
        recall_conversation: Sequence[ChatMessage],
        runtime: FishRuntime,
        tracker: MessagePerformanceTracker,
        *,
        source_message_id: UUID | None = None,
        session_id: UUID | None = None,
        agent_target: AgentResolvedTarget | None = None,
        task_lane_context: TaskLaneContext | None = None,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ) = None,
    ) -> None:
        current_content = _original_current_content(messages)
        runtime.speech_tags = speech_tags_enabled(messages)
        if self._attachments is not None and source_message_id is not None and messages:
            attached = await self._attachments.for_message(source_message_id)
            if attached:
                messages = [
                    *messages[:-1],
                    {**messages[-1], "user_attachments": _chat_attachments(attached)},
                ]
        output_sink = PromotableAnswerSink()
        prepared = self.prepare_initial(
            fish_id,
            messages,
            recall_conversation,
            output_sink,
            tracker,
            agent_target=agent_target,
            task_lane_context=task_lane_context,
            enable_home=(
                (self._home_service is not None or start_home_action is not None)
                and looks_like_home_request(current_content)
            ),
        )
        try:
            await prepared.release_to_public(
                RuntimeAnswerSink(runtime, answer_message_id, tracker),
                source_message_id=source_message_id,
                start_home_action=start_home_action,
                answer_message_id=answer_message_id,
                runtime=runtime,
                session_id=session_id,
            )
        except BaseException:
            await prepared.cancel()
            raise

    def prepare_initial(
        self,
        fish_id: UUID,
        messages: list[ChatMessage],
        recall_conversation: Sequence[ChatMessage],
        sink: PromotableAnswerSink,
        tracker: MessagePerformanceTracker,
        *,
        enable_home: bool,
        agent_target: AgentResolvedTarget | None = None,
        task_lane_context: TaskLaneContext | None = None,
        prepared_reply: PreparedAgentReply | None = None,
        activity_context: str = "",
    ) -> PreparedAnswer:
        """Freeze input once, start eligible lanes, and prepare the selected route."""
        messages = copy.deepcopy(messages)
        speech_tags = speech_tags_enabled(messages)
        sink.configure_speech_tags(speech_tags)
        recall_conversation = tuple(copy.deepcopy(recall_conversation))
        agent_target = copy.deepcopy(agent_target)
        frozen_task_context = copy.deepcopy(
            task_lane_context or TaskLaneContext.empty()
        )
        input_context = LoadedAnswerContext(
            messages=messages,
            recall_conversation=recall_conversation,
            agent_target=agent_target,
            task_lane_context=frozen_task_context,
            activity_context=activity_context,
            speech_tags=speech_tags,
        )
        # Chat picking chooses among reply, recall, search and Agent work; it has
        # no Home choice. A request that looks like Home control takes the route
        # that can carry it out, instead of a reply that only talks about it.
        if (
            self.experiment is not None
            and self.experiment.chat_pick_enabled
            and not enable_home
        ):
            return ExperimentalPreparedAnswer(
                self.experiment, fish_id, input_context, sink, tracker, prepared_reply
            )
        messages = with_speech_tag_rules(messages)
        invocations = self._build_invocations(
            messages,
            enable_home=enable_home,
            agent_target=agent_target,
            task_lane_context=frozen_task_context,
        )
        handles = [
            self._lane_runner.start(invocation, tracker) for invocation in invocations
        ]
        base_messages = {
            _RECALL_LANE_NAME: build_lane_base_messages(
                messages,
                include_memory=True,
                include_environment_image=True,
                recent_limit=self._recent_context_limit,
            ),
            _WEB_LANE_NAME: build_lane_base_messages(
                messages,
                include_memory=False,
                recent_limit=self._recent_context_limit,
            ),
            _HOME_LANE_NAME: build_lane_base_messages(
                messages,
                include_memory=False,
                recent_limit=self._recent_context_limit,
            ),
        }
        route_context = _RouteContext(
            fish_id=fish_id,
            answer_message_id=None,
            recall_conversation=recall_conversation,
            recall_base_messages=base_messages[_RECALL_LANE_NAME],
            web_base_messages=base_messages[_WEB_LANE_NAME],
            home_base_messages=base_messages[_HOME_LANE_NAME],
            runtime=None,
            tracker=tracker,
            output_sink=sink,
            original_query=latest_user_content(messages),
            agent_target=agent_target,
            task_lane_context=frozen_task_context,
            environment_images=_current_environment_images(messages),
        )
        return PreparedInitialAnswer(
            pipeline=self,
            input_hash_builder=lambda: self._initial_input_hash(
                invocations,
                agent_target=agent_target,
                task_lane_context=frozen_task_context,
            ),
            sink=sink,
            input_context=input_context,
            prepared_reply=prepared_reply,
            route_context=route_context,
            handles=handles,
        )

    def _initial_input_hash(
        self,
        invocations: Sequence[AnswerLaneInvocation],
        *,
        agent_target: AgentResolvedTarget | None,
        task_lane_context: TaskLaneContext,
    ) -> str:
        payload = {
            "laneRequests": [
                self._llm.json_stream_request_payload(item.messages)
                for item in invocations
            ],
            "taskFingerprint": {
                "resolvedTargetId": (
                    None if agent_target is None else str(agent_target.target.id)
                ),
                **task_lane_context.fingerprint_payload(),
            },
        }
        return exact_content_hash(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def initial_input_hash(
        self,
        messages: list[ChatMessage],
        *,
        enable_home: bool,
        agent_target: AgentResolvedTarget | None = None,
        task_lane_context: TaskLaneContext | None = None,
    ) -> str:
        """Build a diagnostic payload digest; commit never calls this helper."""
        frozen_task_context = task_lane_context or TaskLaneContext.empty()
        return self._initial_input_hash(
            self._build_invocations(
                messages,
                enable_home=enable_home,
                agent_target=agent_target,
                task_lane_context=frozen_task_context,
            ),
            agent_target=agent_target,
            task_lane_context=frozen_task_context,
        )

    def semantic_initial_input_hash(
        self,
        messages: list[ChatMessage],
        *,
        enable_home: bool,
        agent_target: AgentResolvedTarget | None = None,
        task_lane_context: TaskLaneContext | None = None,
    ) -> str:
        """Build a diagnostic time-independent digest; commit never calls it."""
        invocations = self._build_invocations(
            messages,
            enable_home=enable_home,
            agent_target=agent_target,
            task_lane_context=task_lane_context or TaskLaneContext.empty(),
        )
        payloads = [
            self._llm.json_stream_request_payload(item.messages) for item in invocations
        ]
        return semantic_initial_input_hash(payloads)

    def _build_invocations(
        self,
        messages: list[ChatMessage],
        *,
        enable_home: bool,
        agent_target: AgentResolvedTarget | None,
        task_lane_context: TaskLaneContext,
    ) -> list[AnswerLaneInvocation]:
        messages = with_speech_tag_rules(messages)
        invocations = [
            AnswerLaneInvocation(
                _RECALL_LANE,
                build_recall_lane_messages(
                    messages,
                    recent_limit=self._recent_context_limit,
                ),
            )
        ]
        if self._enable_search:
            invocations.append(
                AnswerLaneInvocation(
                    _WEB_LANE,
                    build_web_lane_messages(
                        messages,
                        recent_limit=self._recent_context_limit,
                    ),
                )
            )
        if (
            self._task_route_service is not None
            and agent_target is not None
            and (
                looks_like_agent_task_request(_original_current_content(messages))
                or task_lane_context.requires_probe
            )
        ):
            invocations.append(
                AnswerLaneInvocation(
                    _task_lane(len(task_lane_context.candidates)),
                    build_task_lane_messages(
                        messages,
                        task_lane_context,
                        recent_limit=self._recent_context_limit,
                    ),
                )
            )
        if enable_home:
            invocations.append(
                AnswerLaneInvocation(
                    _HOME_LANE,
                    build_home_lane_messages(
                        messages,
                        recent_limit=self._recent_context_limit,
                    ),
                )
            )
        return invocations

    async def _prepare_route(
        self, context: _RouteContext, result: AnswerLaneResult
    ) -> None:
        handler = self._route_handlers.get(result.name)
        if handler is None:
            raise RuntimeError(f"selected answer lane has no handler: {result.name}")
        await handler(context, result)

    async def _prepare_home_route(
        self, context: _RouteContext, result: AnswerLaneResult
    ) -> None:
        if not result.accepted or result.home_intent is None:
            raise RuntimeError("selected Home lane is missing its intent")
        if self._home_service is None:
            # Compatibility callers may supply only a post-commit Home callback.
            return
        context.prepared_home = await self._home_service.prepare(
            context.original_query,
            intent=result.home_intent,
            history=context.recall_conversation,
            tracker=context.tracker,
        )
        outcome = context.prepared_home.outcome
        if outcome is not None:
            await self._complete_home_outcome(context, result, outcome)

    async def _prepare_task_route(
        self, context: _RouteContext, result: AnswerLaneResult
    ) -> None:
        if not result.accepted or result.task_kind is None:
            raise RuntimeError("selected Task lane is missing its decision")
        if self._task_route_service is None or context.agent_target is None:
            raise RuntimeError("selected Task lane has no route service or Target")
        context.prepared_task = await self._task_route_service.prepare_selected_route(
            fish_id=context.fish_id,
            resolved_target=context.agent_target,
            task_context=context.task_lane_context,
            kind=result.task_kind,
            candidate_indexes=result.candidate_indexes,
            content=context.original_query,
        )

    async def _commit_route(
        self,
        context: _RouteContext,
        result: AnswerLaneResult,
        *,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ),
    ) -> None:
        if result.name == _TASK_LANE_NAME:
            if self._task_route_service is None or context.prepared_task is None:
                raise RuntimeError("selected Task lane has no prepared route")
            if context.source_message_id is None or context.answer_message_id is None:
                raise RuntimeError("selected Task lane has no durable Message identity")
            await self._task_route_service.commit_prepared_route(
                context.prepared_task,
                source_message_id=context.source_message_id,
                confirmation_message_id=context.answer_message_id,
            )
        elif result.name == _HOME_LANE_NAME:
            await self._commit_home_route(context, result, start_home_action)

    async def _commit_home_route(
        self,
        context: _RouteContext,
        result: AnswerLaneResult,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ),
    ) -> None:
        if result.home_intent is None:
            raise RuntimeError("selected Home lane has no intent")
        if context.prepared_home is not None:
            if self._home_service is None or context.session_id is None:
                raise RuntimeError("prepared Home route has no service or session")
            if context.source_message_id is None:
                raise RuntimeError(
                    "prepared Home route has no durable Message identity"
                )
            home_run = self._home_service.start_prepared(
                context.fish_id,
                context.session_id,
                context.source_message_id,
                context.prepared_home,
                origin_ns=(
                    context.runtime.performance_tracker.origin_ns
                    if context.runtime is not None
                    and context.runtime.performance_tracker is not None
                    else context.tracker.origin_ns
                ),
            )
            # A prepared query already has its outcome and answer. Control is the
            # only route whose actual device result cannot exist before commit.
            await home_run.result()
            return
        if start_home_action is None:
            return
        try:
            home_run = start_home_action(result.home_intent)
        except Exception:
            logger.exception("Failed to start the selected Home route")
            return
        if home_run is not None:
            await self._complete_home_outcome(context, result, await home_run.result())

    async def _complete_home_outcome(
        self,
        context: _RouteContext,
        result: AnswerLaneResult,
        outcome: HomeRunOutcome,
    ) -> None:
        if result.home_intent is None or outcome.intent is not result.home_intent:
            raise RuntimeError("Home outcome does not match the selected H intent")
        if outcome.intent is not HomeIntentType.QUERY:
            return
        continuation = build_home_query_continuation_messages(
            context.home_base_messages,
            context.output_sink.snapshot().content,
            outcome,
        )
        _attach_continuation_environment_images(
            continuation, context.environment_images
        )
        await self._append_continuation(context, continuation)

    async def _complete_web_route(
        self,
        context: _RouteContext,
        result: AnswerLaneResult,
    ) -> None:
        if not result.accepted or result.query is None:
            raise RuntimeError("selected Web lane is missing its query")
        search_context = ""
        try:
            search_context = await self._run_web_query(
                context.web_base_messages,
                result.query,
                context.tracker,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Web search failed; continuing without search context")
        try:
            await self._continue_with_context(
                context,
                context.web_base_messages,
                recall_result=None,
                search_context=search_context,
                environment_images=context.environment_images,
            )
        except _LlmStreamError as error:
            logger.exception("Failed to continue the web-search answer")
            raise AnswerGenerationError from error

    async def _complete_recall_route(
        self,
        context: _RouteContext,
        result: AnswerLaneResult,
    ) -> None:
        self._record_recall_depth(result.recall_depth, context.tracker)
        await context.output_sink.set_end(result.end_conversation)
        if result.recall_depth == 0:
            return
        if result.query is None:
            raise RuntimeError("Recall lane depth requires a query")
        recalled: RecallResult | None = None
        if self._recall_service is not None:
            resolution = await self._resolve_recall_query(
                context.fish_id,
                context.original_query,
                result.query,
                context.tracker,
            )
            recalled = await self._await_recall(
                asyncio.create_task(
                    self._run_recall(
                        context.fish_id,
                        resolution,
                        result.recall_depth,
                        context.recall_conversation,
                        context.tracker,
                    )
                )
            )
        try:
            await self._continue_with_context(
                context,
                context.recall_base_messages,
                recall_result=recalled,
                search_context="",
                environment_images=context.environment_images,
            )
        except _LlmStreamError as error:
            logger.exception("Failed to continue the recalled answer")
            raise AnswerGenerationError from error

    async def _resolve_recall_query(
        self,
        fish_id: UUID,
        original_query: str,
        generated_query: str,
        tracker: MessagePerformanceTracker,
    ) -> RQueryResolution:
        if self._rquery_integrity_service is None:
            return RQueryResolution(
                original_query=original_query,
                generated_query=generated_query,
                final_query=generated_query,
                original_entity_count=0,
                generated_entity_count=0,
                missing_aliases=(),
                outcome="generated",
            )
        return await self._rquery_integrity_service.resolve(
            fish_id,
            original_query=original_query,
            generated_query=generated_query,
            tracker=tracker,
        )

    async def _publish_selected_lane(
        self,
        lane: AnswerLaneHandle,
        output_sink: PromotableAnswerSink,
        *,
        owns: Callable[[], bool] | None = None,
    ) -> None:
        emotion = await lane.emotion
        if owns is not None and not owns():
            return
        await output_sink.set_emotion(emotion)
        while True:
            output = await lane.answers.get()
            if output is None:
                return
            if owns is not None and not owns():
                return
            if output.flush:
                await output_sink.flush()
                continue
            if not output.text:
                continue
            await output_sink.append(output.text)

    async def _run_web_query(
        self,
        messages: list[ChatMessage],
        query: str,
        tracker: MessagePerformanceTracker,
    ) -> str:
        search_messages = build_web_search_context_messages(messages, question=query)
        raw = await self._collect_web_search(
            search_messages,
            tracker,
            step_name="llm.web_search.context",
        )
        return parse_search_context(raw)

    async def _continue_with_context(
        self,
        context: _RouteContext,
        messages: list[ChatMessage],
        *,
        recall_result: RecallResult | None,
        search_context: str,
        environment_images: tuple[ChatImageAttachment, ...],
    ) -> None:
        initial_content = context.output_sink.snapshot().content
        if recall_result is None:
            continuation = self._build_continuation_messages(
                messages, initial_content, recall_result, search_context=search_context
            )
        else:
            with context.tracker.measure("recall.context_build"):
                continuation = self._build_continuation_messages(
                    messages,
                    initial_content,
                    recall_result,
                    search_context=search_context,
                )
        _attach_continuation_environment_images(continuation, environment_images)
        await self._append_continuation(context, continuation)

    async def _append_continuation(
        self, context: _RouteContext, messages: list[ChatMessage]
    ) -> None:
        append = await self._collect_continuation_append(
            messages, context.tracker, require_nonempty=True
        )
        current = context.output_sink.snapshot().content
        joined = _join_answer_append(current, append)
        await context.output_sink.append(joined[len(current) :], continuation=True)

    @staticmethod
    def _record_route_decision(tracker: MessagePerformanceTracker) -> None:
        now_ns = tracker.now_ns()
        tracker.record(
            "answer.decision_ready",
            started_ns=tracker.origin_ns,
            finished_ns=now_ns,
        )

    @staticmethod
    def _record_recall_depth(
        depth: int,
        tracker: MessagePerformanceTracker,
    ) -> None:
        now_ns = tracker.now_ns()
        payload: dict[str, JsonValue] = {
            "recall_depth": depth,
            "context_complete": depth == 0,
        }
        tracker.record(
            "answer.recall_depth",
            started_ns=now_ns,
            finished_ns=now_ns,
            output_payload=payload,
        )


def _original_current_content(messages: Sequence[ChatMessage]) -> str:
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("messages must end with current user content")
    return messages[-1]["content"]


def _current_environment_images(
    messages: Sequence[ChatMessage],
) -> tuple[ChatImageAttachment, ...]:
    if not messages:
        return ()
    return tuple(image.copy() for image in get_environment_images(messages[-1]))


def _attach_continuation_environment_images(
    messages: list[ChatMessage],
    images: tuple[ChatImageAttachment, ...],
) -> None:
    if not images or not messages:
        return
    if any(get_environment_images(message) for message in messages):
        return
    messages[-1]["environment_images"] = tuple(image.copy() for image in images)
    messages[-1]["environment_image"] = images[-1].copy()


def _chat_attachments(
    attached: Sequence[MessageAttachment],
) -> tuple[ChatUserAttachment, ...]:
    return tuple(
        ChatUserAttachment(kind="image", name=item.name, data_url=item.data_url)
        if item.kind == "image"
        else ChatUserAttachment(kind="text", name=item.name, text=item.text or "")
        for item in attached
    )
