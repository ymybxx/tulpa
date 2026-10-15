"""Private Agent replies that finish without generating an assistant answer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from fish_memory.agent.replies import (
    PreparedAgentReply,
    reply_context_fingerprint,
)
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.lane_runtime import AnswerLaneResult
from fish_memory.answer.output_sink import AnswerOutputSink, PromotableAnswerSink
from fish_memory.answer.prepared_answer import HomeRunResultHandle
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.voice_draft_types import exact_content_hash
from fish_memory.home.models import HomeIntentType
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker


class PreparedReplyTurn:
    """An already-classified reply; the API owns its transactional commit.

    Handled replies have no assistant output, so they need neither lane calls
    nor an answer context load. They still share revision, cancellation, and
    readiness tracking with prepared answers.
    """

    def __init__(self, reply: PreparedAgentReply) -> None:
        if reply.requires_answer:
            raise ValueError("a reply requiring an answer must prepare answer lanes")
        self.prepared_reply: PreparedAgentReply | None = reply
        self.sink = PromotableAnswerSink()
        self.input_context = LoadedAnswerContext(messages=[], recall_conversation=())
        self._input_hash = exact_content_hash(
            reply_context_fingerprint(reply.snapshot.context) + "\n" + reply.content
        )
        self._cancelled = False
        self._cleanup: asyncio.Task[None] | None = None

    @property
    def input_hash(self) -> str:
        return self._input_hash

    async def input_ready(self) -> str:
        self._require_owned()
        return self.input_hash

    async def initial_ready(self) -> AnswerLaneResult:
        self._require_owned()
        return AnswerLaneResult(name="reply", accepted=True)

    def initial_done(self) -> bool:
        return True

    async def preparation_ready(self) -> AnswerLaneResult:
        return await self.initial_ready()

    def preparation_done(self) -> bool:
        return True

    def output_token_usage(self) -> int | None:
        return None

    def route_evidence(self) -> dict[str, JsonValue]:
        return {"route": "reply", "context_complete": True}

    def record_preparation_performance(
        self, tracker: MessagePerformanceTracker
    ) -> None:
        # This reply performs no LLM or answer-context computation.
        pass

    def cancel(self) -> asyncio.Task[None]:
        self._cancelled = True
        self.sink.revoke()
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._settle())
        return self._cleanup

    async def _settle(self) -> None:
        pass

    async def release_to_public(
        self,
        public_sink: AnswerOutputSink,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        session_id: UUID | None = None,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ) = None,
    ) -> None:
        raise RuntimeError("a handled Agent reply has no assistant answer to publish")

    def _require_owned(self) -> None:
        if self._cancelled:
            raise RuntimeError("prepared Agent reply was cancelled")
