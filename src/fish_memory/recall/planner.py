from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.llm.logs import logged_call
from fish_memory.recall.models import (
    PlannedRecall,
    RecallQueryPlan,
    RecallSourceRole,
    RecallTimeRange,
)
from fish_memory.recall.prompt import build_recall_plan_messages


class _RecallPlannerOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_alias=True,
        validate_by_name=False,
    )

    source_role: RecallSourceRole = Field(alias="sourceRole")
    entity_hints: tuple[str, ...] = Field(
        alias="entityHints",
        default_factory=tuple,
    )
    time: RecallTimeRange

    @model_validator(mode="before")
    @classmethod
    def reject_non_wire_keys(cls, value: object) -> object:
        if isinstance(value, dict) and set(value).difference(
            {"sourceRole", "entityHints", "time"}
        ):
            raise ValueError("planner options must use wire-format field names")
        return value


class RecallPlanner:
    def __init__(self, client: JsonCompletionClient) -> None:
        self._client = client

    @logged_call("recall.plan")
    async def plan(
        self,
        content: str,
        current_time: datetime,
        *,
        conversation_context: Sequence[ChatMessage] = (),
    ) -> PlannedRecall:
        messages = build_recall_plan_messages(
            content,
            current_time,
            conversation_context=conversation_context,
        )
        request_payload = self._client.json_request_payload(messages)
        completion = await self._client.complete_json(messages)
        options = _RecallPlannerOptions.model_validate_json(completion.content)
        return PlannedRecall(
            plan=RecallQueryPlan(
                semanticQuery=content,
                sourceRole=options.source_role,
                entityHints=options.entity_hints,
                time=options.time,
            ),
            request_payload=request_payload,
            response_payload={
                "content": completion.content,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
        )
