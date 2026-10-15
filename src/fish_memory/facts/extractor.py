import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from fish_memory.facts.models import ExtractedFact, FactExtractionContext
from fish_memory.facts.prompt import build_fact_extraction_messages
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.llm.logs import logged_call
from fish_memory.models import Message
from fish_memory.performance.models import JsonValue


class FactItemOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what: str = Field(min_length=1, max_length=2000)
    where: list[str]
    who: list[str]
    occurred_start: AwareDatetime | None = Field(alias="occurredStart")
    occurred_end: AwareDatetime | None = Field(alias="occurredEnd")
    text: str = Field(min_length=1, max_length=2000)
    entities: list[str] = Field(default_factory=list)

    @field_validator("what", "text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fact value must not be blank")
        return value

    @field_validator("where", "who", "entities")
    @classmethod
    def normalize_string_values(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            name = value.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.occurred_end is not None and self.occurred_start is None:
            raise ValueError("occurredEnd requires occurredStart")
        if (
            self.occurred_start is not None
            and self.occurred_end is not None
            and self.occurred_end < self.occurred_start
        ):
            raise ValueError("occurredEnd must not precede occurredStart")
        return self


class FactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[FactItemOutput]


@dataclass(frozen=True, slots=True)
class FactExtractionResult:
    facts: tuple[ExtractedFact, ...]
    request_payload: JsonValue
    response_payload: JsonValue


class FactExtractor:
    def __init__(self, client: JsonCompletionClient) -> None:
        self._client = client

    @logged_call("memory.extract")
    async def extract(
        self,
        message: Message,
        content: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        messages = build_fact_extraction_messages(message, content, context)
        request_payload = self._client.json_request_payload(messages)
        completion = await self._client.complete_json(messages)
        output = FactOutput.model_validate(
            _normalize_fact_output_payload(completion.content)
        )
        facts = tuple(
            ExtractedFact(
                text=item.text,
                occurred_start=item.occurred_start,
                occurred_end=item.occurred_end,
                entities=tuple(item.entities),
                what=item.what,
                where=tuple(item.where),
                who=tuple(item.who),
            )
            for item in output.facts
        )
        return FactExtractionResult(
            facts=facts,
            request_payload=request_payload,
            response_payload={
                "content": completion.content,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
        )


def _normalize_fact_output_payload(content: str) -> object:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(payload, dict):
        return payload
    facts = payload.get("facts")
    if not isinstance(facts, list):
        return payload
    normalized_facts: list[object] = []
    changed = False
    for item in facts:
        normalized = _normalize_fact_item(item)
        changed = changed or normalized is not item
        normalized_facts.append(normalized)
    if not changed:
        return payload
    return {**payload, "facts": normalized_facts}


def _normalize_fact_item(item: object) -> object:
    if not isinstance(item, Mapping) or len(item) != 1:
        return item
    key = next(iter(item))
    if not isinstance(key, str) or not key.lstrip().startswith("{"):
        return item
    try:
        decoded = json.loads(key)
    except json.JSONDecodeError:
        return item
    if isinstance(decoded, dict):
        return decoded
    return item
