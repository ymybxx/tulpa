import json
from collections.abc import Sequence
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.recall.masking import EntityAliases

FISH_ID = UUID(int=1)
WOLF_ID = UUID(int=2)
SWORD_ID = UUID(int=3)


class FakeRQueryStore:
    def __init__(
        self,
        responses: Sequence[tuple[tuple[EntityAliases, ...], ...] | Exception],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[UUID, tuple[str, ...]]] = []

    async def resolve_query_entities_batch(
        self,
        fish_id: UUID,
        queries: Sequence[str],
    ) -> tuple[tuple[EntityAliases, ...], ...]:
        self.calls.append((fish_id, tuple(queries)))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeJsonClient:
    def __init__(self, completions: Sequence[JsonCompletion]) -> None:
        self._completions = list(completions)
        self.messages: list[Sequence[ChatMessage]] = []

    async def complete_json(self, messages: Sequence[ChatMessage]) -> JsonCompletion:
        self.messages.append(messages)
        return self._completions.pop(0)

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": list(messages)}


class PayloadErrorJsonClient(FakeJsonClient):
    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        del messages
        raise RuntimeError("payload unavailable")


def entity(entity_id: UUID, *aliases: str) -> EntityAliases:
    return EntityAliases(entity_id, aliases)


def completion(query: str) -> JsonCompletion:
    return JsonCompletion(
        content='{"q":"' + query + '"}',
        input_tokens=7,
        output_tokens=3,
    )


def step_names(tracker: MessagePerformanceTracker) -> list[str]:
    return [step.name for step in tracker.snapshot()]


@pytest.mark.asyncio
async def test_alias_change_with_same_entity_needs_no_repair() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    store = FakeRQueryStore(
        [
            (
                (entity(WOLF_ID, "杰洛特", "白狼"),),
                (entity(WOLF_ID, "杰洛特", "白狼"),),
            )
        ]
    )
    client = FakeJsonClient([completion("unused")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="杰洛特使用过什么剑",
        tracker=tracker,
    )

    assert result.final_query == "杰洛特使用过什么剑"
    assert result.original_entity_count == 1
    assert result.generated_entity_count == 1
    assert result.missing_aliases == ()
    assert result.outcome == "generated"
    assert client.messages == []
    assert len(store.calls) == 1
    assert step_names(tracker) == ["answer.rquery_validation"]


@pytest.mark.asyncio
async def test_missing_entity_repairs_once_and_accepts_valid_result() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), ()), ((wolf,), (wolf,))])
    client = FakeJsonClient([completion("  白狼使用过什么剑  ")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="他使用过什么剑",
        tracker=tracker,
    )

    assert result.final_query == "白狼使用过什么剑"
    assert result.original_entity_count == 1
    assert result.generated_entity_count == 0
    assert result.missing_aliases == ("白狼",)
    assert result.outcome == "repaired"
    assert len(client.messages) == 1
    assert len(store.calls) == 2
    assert step_names(tracker) == [
        "llm.answer.rquery_repair",
        "answer.rquery_validation",
    ]


@pytest.mark.asyncio
async def test_repair_prompt_only_lists_aliases_mentioned_in_original_query() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), ()), ((wolf,), (wolf,))])
    client = FakeJsonClient([completion("白狼使用过什么剑")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="他使用过什么剑",
        tracker=tracker,
    )

    prompt_payload = json.loads(client.messages[0][1]["content"])
    assert result.missing_aliases == ("白狼",)
    assert prompt_payload["missingAliases"] == ["白狼"]


@pytest.mark.asyncio
async def test_invalid_repair_falls_back_to_original() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), ()), ((wolf,), ())])
    client = FakeJsonClient([completion("他使用过什么剑")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="他使用过什么剑",
        tracker=tracker,
    )

    assert result.final_query == "白狼以前用过什么剑"
    assert result.outcome == "original_fallback"
    assert len(client.messages) == 1
    assert len(store.calls) == 2


@pytest.mark.asyncio
async def test_second_entity_loss_falls_back_without_loop() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    sword = entity(SWORD_ID, "银剑")
    store = FakeRQueryStore([((wolf, sword), (sword,)), ((wolf, sword), (wolf,))])
    client = FakeJsonClient([completion("白狼使用过什么")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼的银剑在哪里",
        generated_query="银剑在哪里",
        tracker=tracker,
    )

    assert result.final_query == "白狼的银剑在哪里"
    assert result.outcome == "original_fallback"
    assert len(client.messages) == 1
    assert len(store.calls) == 2


@pytest.mark.asyncio
async def test_validation_database_error_falls_back_to_original() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    store = FakeRQueryStore([RuntimeError("database unavailable")])
    client = FakeJsonClient([completion("unused")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="杰洛特使用过什么剑",
        tracker=tracker,
    )

    assert result.final_query == "白狼以前用过什么剑"
    assert result.original_entity_count == 0
    assert result.generated_entity_count == 0
    assert result.missing_aliases == ()
    assert result.outcome == "validation_degraded"
    assert client.messages == []
    assert len(store.calls) == 1
    assert step_names(tracker) == ["answer.rquery_validation"]


@pytest.mark.asyncio
async def test_validation_records_elapsed_duration() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    clock_values = iter((0, 1_000, 6_000))
    tracker = MessagePerformanceTracker(clock=lambda: next(clock_values))
    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), (wolf,))])
    client = FakeJsonClient([completion("unused")])

    await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="杰洛特使用过什么剑",
        tracker=tracker,
    )

    validation = tracker.snapshot()[0]
    assert validation.name == "answer.rquery_validation"
    assert validation.duration_us == 5


@pytest.mark.asyncio
async def test_payload_error_records_the_repair_attempt() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), ())])
    client = PayloadErrorJsonClient([completion("unused")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="他使用过什么剑",
        tracker=tracker,
    )

    repair, validation = tracker.snapshot()
    assert result.outcome == "original_fallback"
    assert repair.name == "llm.answer.rquery_repair"
    assert repair.input_payload is None
    assert repair.error == "RuntimeError: payload unavailable"
    assert validation.name == "answer.rquery_validation"
    assert len(store.calls) == 1


@pytest.mark.asyncio
async def test_blank_repair_query_falls_back_without_revalidation() -> None:
    from fish_memory.answer.rquery import RQueryIntegrityService

    wolf = entity(WOLF_ID, "杰洛特", "白狼")
    store = FakeRQueryStore([((wolf,), ())])
    client = FakeJsonClient([completion("   ")])
    tracker = MessagePerformanceTracker()

    result = await RQueryIntegrityService(store, client).resolve(
        FISH_ID,
        original_query="白狼以前用过什么剑",
        generated_query="他使用过什么剑",
        tracker=tracker,
    )

    assert result.final_query == "白狼以前用过什么剑"
    assert result.outcome == "original_fallback"
    assert len(store.calls) == 1
