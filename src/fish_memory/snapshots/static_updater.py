from collections.abc import Sequence
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from fish_memory.llm.client import JsonCompletionClient
from fish_memory.llm.logs import logged_call
from fish_memory.performance.models import JsonValue
from fish_memory.snapshots.models import (
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemState,
    SnapshotPreUpdate,
    SnapshotPreUpdateResult,
)
from fish_memory.snapshots.preupdate_normalization import (
    normalize_preupdate_output,
)
from fish_memory.snapshots.static_prompt import build_static_profile_preupdate_messages
from fish_memory.snapshots.updater import _validate_preupdate_bounds

NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]


class _SourceIndexesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_fact_indexes: list[NonNegativeStrictInt] = Field(
        alias="sourceFactIndexes",
        min_length=1,
    )

    @field_validator("source_fact_indexes")
    @classmethod
    def reject_duplicate_indexes(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("source fact indexes must not contain duplicates")
        return values


class _PreCreateOutput(_SourceIndexesOutput):
    summary: str = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("static profile summary must not be blank")
        return value


class _PreUpdateMarkerOutput(_SourceIndexesOutput):
    item_index: NonNegativeStrictInt = Field(alias="itemIndex")


class _StaticPreUpdateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creates: list[_PreCreateOutput] = Field(default_factory=list)
    updates: list[_PreUpdateMarkerOutput] = Field(default_factory=list)


class StaticProfileUpdater:
    def __init__(self, client: JsonCompletionClient) -> None:
        self._client = client

    @logged_call("memory.static")
    async def preupdate(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
    ) -> SnapshotPreUpdateResult:
        messages = build_static_profile_preupdate_messages(
            existing_items,
            new_facts,
            current_time,
        )
        request_payload = self._client.json_request_payload(messages)
        completion = await self._client.complete_json(messages)
        normalized = normalize_preupdate_output(
            completion.content,
            item_count=len(existing_items),
            fact_count=len(new_facts),
        )
        output = (
            _StaticPreUpdateOutput.model_validate(normalized.payload)
            if normalized.parsed
            else _StaticPreUpdateOutput.model_validate_json(completion.content)
        )
        pre_update = SnapshotPreUpdate(
            creates=tuple(
                PreCreateOperation(
                    summary=item.summary,
                    source_fact_indexes=tuple(item.source_fact_indexes),
                )
                for item in output.creates
            ),
            updates=tuple(
                PreUpdateMarker(
                    item_index=item.item_index,
                    source_fact_indexes=tuple(item.source_fact_indexes),
                )
                for item in output.updates
            ),
        )
        _validate_preupdate_bounds(
            pre_update,
            item_count=len(existing_items),
            fact_count=len(new_facts),
        )
        response_payload: dict[str, JsonValue] = {
            "content": completion.content,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        }
        if normalized.dropped_operations:
            response_payload["dropped_operations"] = list(normalized.dropped_operations)
        if normalized.normalizations:
            response_payload["normalizations"] = list(normalized.normalizations)
        return SnapshotPreUpdateResult(
            pre_update=pre_update,
            request_payload=request_payload,
            response_payload=response_payload,
        )
