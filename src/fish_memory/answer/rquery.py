from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from fish_memory.answer.rquery_prompt import build_rquery_repair_messages
from fish_memory.entities.normalization import normalize_entity_alias
from fish_memory.llm.client import JsonCompletion, JsonCompletionClient
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.recall.masking import EntityAliases


@dataclass(frozen=True, slots=True)
class RQueryResolution:
    original_query: str
    generated_query: str
    final_query: str
    original_entity_count: int
    generated_entity_count: int
    missing_aliases: tuple[str, ...]
    outcome: Literal[
        "generated",
        "repaired",
        "original_fallback",
        "validation_degraded",
    ]


class RQueryEntityStore(Protocol):
    async def resolve_query_entities_batch(
        self,
        fish_id: UUID,
        queries: Sequence[str],
    ) -> tuple[tuple[EntityAliases, ...], ...]: ...


class _RepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1)


class RQueryIntegrityService:
    def __init__(
        self,
        store: RQueryEntityStore,
        client: JsonCompletionClient,
    ) -> None:
        self._store = store
        self._client = client

    async def resolve(
        self,
        fish_id: UUID,
        *,
        original_query: str,
        generated_query: str,
        tracker: MessagePerformanceTracker,
    ) -> RQueryResolution:
        validation_started_ns = tracker.now_ns()
        original_entities: tuple[EntityAliases, ...] = ()
        generated_entities: tuple[EntityAliases, ...] = ()
        missing_aliases: tuple[str, ...] = ()
        validation_error: str | None = None
        try:
            (
                original_entities,
                generated_entities,
            ) = await self._store.resolve_query_entities_batch(
                fish_id,
                (original_query, generated_query),
            )
            missing_aliases = _missing_aliases(
                original_query,
                original_entities,
                generated_entities,
            )
        except Exception as error:
            validation_error = _error_text(error)
            result = RQueryResolution(
                original_query=original_query,
                generated_query=generated_query,
                final_query=original_query,
                original_entity_count=0,
                generated_entity_count=0,
                missing_aliases=(),
                outcome="validation_degraded",
            )
            _record_validation(
                tracker,
                result,
                started_ns=validation_started_ns,
                error=validation_error,
            )
            return result

        if not missing_aliases:
            result = RQueryResolution(
                original_query=original_query,
                generated_query=generated_query,
                final_query=generated_query,
                original_entity_count=len(original_entities),
                generated_entity_count=len(generated_entities),
                missing_aliases=(),
                outcome="generated",
            )
            _record_validation(
                tracker,
                result,
                started_ns=validation_started_ns,
            )
            return result

        repaired_query: str | None = None
        try:
            repaired_query = await self._repair_query(
                original_query,
                generated_query,
                missing_aliases,
                tracker,
            )
        except Exception:
            result = _original_fallback(
                original_query,
                generated_query,
                original_entities,
                generated_entities,
                missing_aliases,
            )
            _record_validation(
                tracker,
                result,
                started_ns=validation_started_ns,
            )
            return result

        try:
            (
                original_recheck,
                repaired_entities,
            ) = await self._store.resolve_query_entities_batch(
                fish_id,
                (original_query, repaired_query),
            )
            if _missing_aliases(
                original_query,
                original_recheck,
                repaired_entities,
            ):
                result = _original_fallback(
                    original_query,
                    generated_query,
                    original_entities,
                    generated_entities,
                    missing_aliases,
                )
            else:
                result = RQueryResolution(
                    original_query=original_query,
                    generated_query=generated_query,
                    final_query=repaired_query,
                    original_entity_count=len(original_entities),
                    generated_entity_count=len(generated_entities),
                    missing_aliases=missing_aliases,
                    outcome="repaired",
                )
        except Exception as error:
            result = RQueryResolution(
                original_query=original_query,
                generated_query=generated_query,
                final_query=original_query,
                original_entity_count=len(original_entities),
                generated_entity_count=len(generated_entities),
                missing_aliases=missing_aliases,
                outcome="validation_degraded",
            )
            validation_error = _error_text(error)

        _record_validation(
            tracker,
            result,
            started_ns=validation_started_ns,
            error=validation_error,
        )
        return result

    async def _repair_query(
        self,
        original_query: str,
        generated_query: str,
        missing_aliases: tuple[str, ...],
        tracker: MessagePerformanceTracker,
    ) -> str:
        started_ns = tracker.now_ns()
        completion: JsonCompletion | None = None
        request_payload: JsonValue = None
        error: str | None = None
        try:
            messages = build_rquery_repair_messages(
                original_query,
                generated_query,
                missing_aliases,
                "generated query omitted one or more original entities",
            )
            request_payload = self._client.json_request_payload(messages)
            completion = await self._client.complete_json(messages)
            query = _RepairOutput.model_validate_json(completion.content).q.strip()
            if not query:
                raise ValueError("repair query must not be blank")
            return query
        except Exception as exception:
            error = _error_text(exception)
            raise
        finally:
            tracker.record(
                "llm.answer.rquery_repair",
                started_ns=started_ns,
                finished_ns=tracker.now_ns(),
                input_payload=request_payload,
                output_payload=_completion_payload(completion),
                error=error,
            )


def _missing_aliases(
    original_query: str,
    original_entities: Sequence[EntityAliases],
    candidate_entities: Sequence[EntityAliases],
) -> tuple[str, ...]:
    candidate_ids = {entity.entity_id for entity in candidate_entities}
    normalized_original_query = normalize_entity_alias(original_query)
    return tuple(
        alias
        for entity in original_entities
        if entity.entity_id not in candidate_ids
        for alias in entity.aliases
        if normalize_entity_alias(alias) in normalized_original_query
    )


def _original_fallback(
    original_query: str,
    generated_query: str,
    original_entities: Sequence[EntityAliases],
    generated_entities: Sequence[EntityAliases],
    missing_aliases: tuple[str, ...],
) -> RQueryResolution:
    return RQueryResolution(
        original_query=original_query,
        generated_query=generated_query,
        final_query=original_query,
        original_entity_count=len(original_entities),
        generated_entity_count=len(generated_entities),
        missing_aliases=missing_aliases,
        outcome="original_fallback",
    )


def _completion_payload(completion: JsonCompletion | None) -> JsonValue:
    if completion is None:
        return None
    return {
        "content": completion.content,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
    }


def _record_validation(
    tracker: MessagePerformanceTracker,
    result: RQueryResolution,
    *,
    started_ns: int,
    error: str | None = None,
) -> None:
    tracker.record(
        "answer.rquery_validation",
        started_ns=started_ns,
        finished_ns=tracker.now_ns(),
        output_payload={
            "outcome": result.outcome,
            "originalEntityCount": result.original_entity_count,
            "generatedEntityCount": result.generated_entity_count,
            "missingAliases": list(result.missing_aliases),
        },
        error=error,
    )


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
