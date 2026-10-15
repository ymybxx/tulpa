"""One private revision: Jev and forced candidate wording run concurrently."""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fish_memory.answer.context import get_environment_images
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.lane_runtime import AnswerLaneResult
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.voice_draft_types import exact_content_hash
from fish_memory.experimental.channels import (
    ChannelResult,
    generate_channel,
    render_state,
)
from fish_memory.experimental.context import (
    SPEECH_THOUGHT_CONTEXT,
    expression_context,
    with_recent_thoughts,
)
from fish_memory.experimental.perception import PerceptionInput

if TYPE_CHECKING:
    from fish_memory.experimental.service import ExperimentalService


class ExperimentalPreparedAnswer:
    def __init__(
        self,
        service: ExperimentalService,
        fish_id: UUID,
        context: LoadedAnswerContext,
        sink: PromotableAnswerSink,
        tracker: Any,
        prepared_reply: Any = None,
    ):
        self.service, self.fish_id = service, fish_id
        self.input_context, self.sink = copy.deepcopy(context), sink
        self.sink.configure_speech_tags(context.speech_tags)
        self.prepared_reply, self.tracker = prepared_reply, tracker
        self.revision = service.revision
        captured_at = service.now()
        recent = with_recent_thoughts(context.messages, SPEECH_THOUGHT_CONTEXT)
        trigger = "用户已经说完末尾这句话，现在决定如何回应。"
        if any(message.get("user_attachments") for message in context.messages):
            # History only names what was sent. Told nothing more, Jev handed
            # a sent file to the agent to read 10 of 10 times; told the reply
            # sees it, it answered directly 10 of 10.
            trigger += "用户这次发来的图片和文件，内容直接可见。"
        self.state = render_state(recent, captured_at, trigger)
        self.expression_context = expression_context(
            with_recent_thoughts(self.input_context.messages, SPEECH_THOUGHT_CONTEXT),
            captured_at,
            trigger,
        )
        self.input_hash = exact_content_hash(self.state + str(self.revision))
        self.result: AnswerLaneResult | None = None
        self._channel: ChannelResult | None = None
        self._selected = asyncio.Event()
        self._cancelled = False
        self._released = False
        self._cleanup: asyncio.Task[None] | None = None
        self._candidates: dict[
            str, tuple[PromotableAnswerSink, asyncio.Task[ChannelResult]]
        ] = {}
        self._perception_task = (
            asyncio.create_task(
                service.perception.capture(fish_id, captured_at, model=service.chat)
            )
            if service.perception is not None
            else None
        )
        if self._perception_task is not None:
            self._perception_task.add_done_callback(_observe)
        self._task = asyncio.create_task(
            self._prepare(), name="experimental-user-preparation"
        )
        self._task.add_done_callback(_observe)
        service.preparations.add(self)
        if service.runtime is not None:
            service.runtime.refresh_busy()

    @property
    def pending_input(self) -> bool:
        return not self._released and not self._cancelled

    def _owned(self):
        if (
            self._cancelled
            or self.revision != self.service.revision
            or not self.service.chat_pick_enabled
        ):
            raise asyncio.CancelledError

    async def _prepare(self) -> AnswerLaneResult:
        try:
            available = await self.service.available(self.input_context)
            self._owned()
            for action in available:
                if action != "wait":
                    sink = PromotableAnswerSink(
                        speech_tags=self.input_context.speech_tags
                    )
                    task = asyncio.create_task(self._generate_candidate(action, sink))
                    task.add_done_callback(_observe)
                    self._candidates[action] = sink, task
            action = await self.service.decide(self.state, available)
            self._owned()
            for name, (sink, task) in self._candidates.items():
                if name != action:
                    sink.revoke()
                    task.cancel()
            if action == "wait":
                self._channel = ChannelResult(action)
            else:
                sink, task = self._candidates[action]
                await sink.attach(self.sink)
                self._selected.set()
                self._channel = await task
            self.result = AnswerLaneResult(
                name=action,
                accepted=True,
                query=self._channel.query,
                end_conversation=self._channel.end,
            )
            return self.result
        finally:
            self._selected.set()
            for _sink, task in self._candidates.values():
                if not task.done():
                    task.cancel()
            if self._perception_task is not None and not self._perception_task.done():
                self._perception_task.cancel()

    async def _generate_candidate(self, action, sink):
        perception = (
            await asyncio.shield(self._perception_task)
            if self._perception_task is not None
            else PerceptionInput(
                images=get_environment_images(self.input_context.messages[-1])
            )
        )
        self._owned()
        return await generate_channel(
            self.service.chat,
            action,
            self.expression_context,
            sink,
            perception=perception,
        )

    async def input_ready(self) -> str:
        return self.input_hash

    async def initial_ready(self) -> AnswerLaneResult:
        return await asyncio.shield(self._task)

    preparation_ready = initial_ready

    def initial_done(self) -> bool:
        return self._task.done()

    preparation_done = initial_done

    def output_token_usage(self) -> int | None:
        return None

    def route_evidence(self):
        return (
            {"mode": "experimental", "action": self.result.name}
            if self.result
            else None
        )

    def record_preparation_performance(self, tracker):
        now = tracker.now_ns()
        tracker.record(
            "experimental.preparation",
            started_ns=now,
            finished_ns=now,
            output_payload=self.route_evidence(),
        )

    async def release_to_public(
        self,
        public_sink,
        *,
        source_message_id,
        answer_message_id,
        runtime,
        session_id=None,
        start_home_action=None,
    ):
        self._owned()
        if self._released:
            raise RuntimeError("Experimental answer already released")
        self._released = True
        if self.service.runtime is not None:
            self.service.runtime.refresh_busy()
        await self._selected.wait()
        self._owned()
        if self._task.done():
            await self.initial_ready()
        await self.sink.attach(public_sink)
        await self.initial_ready()
        self._owned()
        channel = self._channel
        if channel and channel.action in {"r", "t", "w"}:
            if session_id is None or source_message_id is None:
                raise RuntimeError("Selected work requires a committed user message")
            self.service.launch(
                self.fish_id,
                session_id,
                source_message_id,
                channel,
                self.input_context,
                self.state,
                # The person is waiting for the answer: no written recall summary.
                summarize_recall=False,
            )

    def cancel(self) -> asyncio.Task[None]:
        if self._cleanup is not None:
            return self._cleanup
        self._cancelled = True
        if self.service.runtime is not None:
            self.service.runtime.refresh_busy()
        self.sink.revoke()
        self._task.cancel()
        if self._perception_task is not None:
            self._perception_task.cancel()
        for sink, task in self._candidates.values():
            sink.revoke()
            task.cancel()

        async def settle():
            await asyncio.gather(
                self._task,
                *(
                    (self._perception_task,)
                    if self._perception_task is not None
                    else ()
                ),
                *(task for _, task in self._candidates.values()),
                return_exceptions=True,
            )

        self._cleanup = asyncio.create_task(settle())
        return self._cleanup


def _observe(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()
