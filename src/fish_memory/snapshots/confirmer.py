from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.performance.models import JsonValue
from fish_memory.snapshots.confirmation_prompt import (
    build_snapshot_confirmation_messages,
)
from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    SnapshotFact,
    SnapshotItemState,
)
from fish_memory.snapshots.static_confirmation_prompt import (
    build_static_profile_confirmation_messages,
)

NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]


class SnapshotEvidenceError(ValueError):
    """Snapshot confirmation inputs disagree with persisted evidence."""


@dataclass(frozen=True, slots=True)
class SnapshotConfirmationResult:
    operations: tuple[ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore, ...]
    request_payload: JsonValue
    response_payload: JsonValue


class _ConfirmationOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: str = Field(min_length=1, max_length=100)
    new_support_indexes: list[NonNegativeStrictInt] = Field(alias="newSupportIndexes")
    summary: str = Field(min_length=1)

    @field_validator("basis")
    @classmethod
    def reject_blank_basis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mutation basis must not be blank")
        return value

    @field_validator("summary")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("snapshot summary must not be blank")
        return value

    @field_validator("new_support_indexes")
    @classmethod
    def reject_duplicate_new_support_indexes(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("new support indexes must not contain duplicates")
        return values


class _UpdateOutput(_ConfirmationOperation):
    action: Literal["update"]
    item_index: NonNegativeStrictInt = Field(alias="itemIndex")
    remove_support_indexes: list[NonNegativeStrictInt] = Field(
        alias="removeSupportIndexes"
    )

    @field_validator("remove_support_indexes")
    @classmethod
    def reject_duplicate_remove_support_indexes(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("remove support indexes must not contain duplicates")
        return values


class _CreateOutput(_ConfirmationOperation):
    action: Literal["create"]

    @field_validator("new_support_indexes")
    @classmethod
    def require_new_support_indexes(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("new support indexes must not be empty")
        return values


class _IgnoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["ignore"]
    basis: str = Field(min_length=1, max_length=100)
    new_support_indexes: list[NonNegativeStrictInt] = Field(
        alias="newSupportIndexes",
        min_length=1,
    )

    @field_validator("basis")
    @classmethod
    def reject_blank_basis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mutation basis must not be blank")
        return value

    @field_validator("new_support_indexes")
    @classmethod
    def reject_duplicate_new_support_indexes(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("new support indexes must not contain duplicates")
        return values


ConfirmationOperationOutput = Annotated[
    _UpdateOutput | _CreateOutput,
    Field(discriminator="action"),
]
StaticConfirmationOperationOutput = Annotated[
    _UpdateOutput | _CreateOutput | _IgnoreOutput,
    Field(discriminator="action"),
]


class _ConfirmationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operations: list[ConfirmationOperationOutput] = Field(min_length=1)


class _StaticConfirmationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operations: list[StaticConfirmationOperationOutput] = Field(min_length=1)


class SnapshotUpdateConfirmer:
    def __init__(self, client: JsonCompletionClient) -> None:
        self._client = client

    async def confirm(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> SnapshotConfirmationResult:
        _validate_support_facts(item, support_facts)
        _validate_unique_new_fact_ids(new_facts)
        messages = self._build_messages(
            item_index=item_index,
            item=item,
            support_facts=support_facts,
            new_facts=new_facts,
            recent_turns=recent_turns,
        )
        request_payload = self._client.json_request_payload(messages)
        completion = await self._client.complete_json(messages)
        output_operations = self._parse_operations(completion.content)
        operations = _confirmed_operations(
            output_operations,
            item_index,
            support_facts,
            new_facts,
        )
        self._validate_result(operations, new_fact_count=len(new_facts))
        return SnapshotConfirmationResult(
            operations=operations,
            request_payload=request_payload,
            response_payload={
                "content": completion.content,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
        )

    def _build_messages(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> list[ChatMessage]:
        return build_snapshot_confirmation_messages(
            item_index=item_index,
            item=item,
            support_facts=support_facts,
            new_facts=new_facts,
            recent_turns=recent_turns,
        )

    def _parse_operations(
        self,
        content: str,
    ) -> Sequence[_UpdateOutput | _CreateOutput | _IgnoreOutput]:
        return _ConfirmationOutput.model_validate_json(content).operations

    def _validate_result(
        self,
        operations: Sequence[ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore],
        *,
        new_fact_count: int,
    ) -> None:
        del operations, new_fact_count


class StaticProfileUpdateConfirmer(SnapshotUpdateConfirmer):
    """Confirm Static updates under the independent long-term profile contract."""

    def _build_messages(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> list[ChatMessage]:
        return build_static_profile_confirmation_messages(
            item_index=item_index,
            item=item,
            support_facts=support_facts,
            new_facts=new_facts,
            recent_turns=recent_turns,
        )

    def _parse_operations(
        self,
        content: str,
    ) -> Sequence[_UpdateOutput | _CreateOutput | _IgnoreOutput]:
        return _StaticConfirmationOutput.model_validate_json(content).operations

    def _validate_result(
        self,
        operations: Sequence[ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore],
        *,
        new_fact_count: int,
    ) -> None:
        if any(
            isinstance(operation, ConfirmedUpdate) and not operation.new_support_indexes
            for operation in operations
        ):
            raise ValueError("static update must select at least one new fact")
        selected_indexes = tuple(
            index for operation in operations for index in operation.new_support_indexes
        )
        if set(selected_indexes) != set(range(new_fact_count)):
            raise ValueError("static confirmation must classify every new fact")


def _validate_support_facts(
    item: SnapshotItemState,
    support_facts: Sequence[SnapshotFact],
) -> None:
    if tuple(fact.id for fact in support_facts) != item.support_fact_ids:
        raise SnapshotEvidenceError(
            "support facts must match the target item's current support"
        )


def _validate_unique_new_fact_ids(new_facts: Sequence[SnapshotFact]) -> None:
    fact_ids = tuple(fact.id for fact in new_facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise SnapshotEvidenceError("duplicate new fact id")


def _confirmed_operations(
    output_operations: Sequence[_UpdateOutput | _CreateOutput | _IgnoreOutput],
    item_index: int,
    support_facts: Sequence[SnapshotFact],
    new_facts: Sequence[SnapshotFact],
) -> tuple[ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore, ...]:
    operations: list[ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore] = []
    selected_new_fact_ids: set[UUID] = set()
    update_count = 0
    for operation in output_operations:
        _validate_indexes(operation.new_support_indexes, len(new_facts), "new support")
        selected_fact_ids = {
            new_facts[index].id for index in operation.new_support_indexes
        }
        if selected_new_fact_ids.intersection(selected_fact_ids):
            raise ValueError("a new support fact must not be selected more than once")
        selected_new_fact_ids.update(selected_fact_ids)
        if isinstance(operation, _UpdateOutput):
            update_count += 1
            if update_count > 1:
                raise ValueError("a confirmation may contain at most one update")
            if operation.item_index != item_index:
                raise ValueError("update must target its original item index")
            _validate_indexes(
                operation.remove_support_indexes,
                len(support_facts),
                "remove support",
            )
            operations.append(
                ConfirmedUpdate(
                    basis=operation.basis,
                    item_index=operation.item_index,
                    new_support_indexes=tuple(operation.new_support_indexes),
                    remove_support_indexes=tuple(operation.remove_support_indexes),
                    summary=operation.summary,
                )
            )
        elif isinstance(operation, _CreateOutput):
            operations.append(
                ConfirmedCreate(
                    basis=operation.basis,
                    new_support_indexes=tuple(operation.new_support_indexes),
                    summary=operation.summary,
                )
            )
        else:
            operations.append(
                ConfirmedIgnore(
                    basis=operation.basis,
                    new_support_indexes=tuple(operation.new_support_indexes),
                )
            )
    return tuple(operations)


def _validate_indexes(indexes: Sequence[int], upper_bound: int, name: str) -> None:
    if any(index >= upper_bound for index in indexes):
        raise ValueError(f"{name} index is outside its local fact collection")
