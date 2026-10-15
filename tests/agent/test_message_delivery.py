from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.agent.message_delivery import (
    SAFE_AGENT_MESSAGE_FALLBACK,
    AgentMessageComposer,
    AgentMessageProcessor,
)
from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionKind,
    AgentInteractionWorkItem,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    AgentTask,
    ConnectorTypeSlug,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.performance.models import JsonValue
from fish_memory.soul import SoulProfile

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
RUN_ID = UUID("00000000-0000-0000-0000-000000000010")
INTERACTION_ID = UUID("00000000-0000-0000-0000-000000000020")
DEFAULT_SOUL = SoulProfile("旅行伙伴", "温柔又简洁", "小鱼")


def run_work(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    output: str | None = "规范化完成结果",
    provider_data: Mapping[str, object] | None = None,
) -> AgentRunWorkItem:
    endpoint = AgentEndpoint(
        id=UUID(int=2),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="https://provider.invalid/private-endpoint",
        created_at=NOW,
        updated_at=NOW,
    )
    target = AgentTarget(
        id=UUID(int=3),
        endpoint_id=endpoint.id,
        external_key="external-target-secret-id",
        route_prefix="/profile/private",
        credential_ref="credential:private-ref",
        connector_config={"raw_tool_stream": "never prompt this"},
        display_name="Provider target",
        enabled=True,
        created_at=NOW,
        updated_at=NOW,
    )
    task = AgentTask(
        id=UUID(int=4),
        fish_id=FISH_ID,
        session_id=UUID(int=40),
        target_id=target.id,
        external_session_id="external-session-private-id",
        title="整理旅行计划",
        created_at=NOW,
        updated_at=NOW,
    )
    run = AgentRun(
        id=RUN_ID,
        task_id=task.id,
        target_id=target.id,
        fish_id=task.fish_id,
        session_id=task.session_id,
        previous_run_id=None,
        source_message_id=UUID(int=5),
        confirmation_message_id=None,
        input="raw task input must not enter the composer",
        external_run_id="external-run-private-id",
        history_source=AgentHistorySource.NONE,
        status=status,
        output=output,
        result_message_id=None,
        error_code="provider_error",
        error_message="raw provider failure private-token",
        provider_data=provider_data,
        created_at=NOW,
        started_at=NOW,
        updated_at=NOW,
        finished_at=NOW,
    )
    return AgentRunWorkItem(run=run, task=task, target=target, endpoint=endpoint)


def interaction_work(
    *,
    kind: AgentInteractionKind = AgentInteractionKind.APPROVAL,
    provider_data: Mapping[str, object] | None = None,
) -> AgentInteractionWorkItem:
    base = run_work(status=AgentRunStatus.WAITING_USER, output="需要用户提供信息")
    interaction = AgentInteraction(
        id=INTERACTION_ID,
        run_id=base.run.id,
        fish_id=base.run.fish_id,
        session_id=base.run.session_id,
        kind=kind,
        request_message_id=None,
        response_message_id=None,
        next_run_id=None,
        provider_data=provider_data
        or {
            "command": "deploy safe-app",
            "description": "发布已经审核的版本",
            "allowed_decisions": ["once", "deny"],
            "request_id": "external-request-private-id",
            "raw_tool_stream": "private tool output",
        },
        created_at=NOW,
        resolved_at=None,
        closed_at=None,
    )
    return AgentInteractionWorkItem(
        interaction=interaction,
        run=base.run,
        task=base.task,
        target=base.target,
        endpoint=base.endpoint,
    )


class FakeCompletionClient:
    def __init__(self, content: str | BaseException) -> None:
        self.content = content
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(self, messages: Sequence[ChatMessage]) -> JsonCompletion:
        self.calls.append(messages)
        if isinstance(self.content, BaseException):
            raise self.content
        return JsonCompletion(self.content)

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": list(messages)}


class FakeSoulLoader:
    def __init__(
        self,
        profile: SoulProfile | None = DEFAULT_SOUL,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.profile = profile
        self.error = error
        self.calls: list[UUID] = []

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        self.calls.append(fish_id)
        if self.error is not None:
            raise self.error
        return self.profile


@pytest.mark.asyncio
async def test_composer_accepts_only_exact_message_contract_and_safe_prompt() -> None:
    client = FakeCompletionClient('{"m":"happy","a":"旅行计划整理好了。"}')
    soul = FakeSoulLoader()
    composer = AgentMessageComposer(client, soul)

    draft = await composer.compose_run(run_work())

    assert draft.emotion == "happy"
    assert draft.content == "旅行计划整理好了。"
    assert draft.used_fallback is False
    assert soul.calls == [FISH_ID]
    prompt = json.dumps(client.calls[0], ensure_ascii=False)
    assert "旅行伙伴" in prompt
    assert "温柔又简洁" in prompt
    assert "整理旅行计划" in prompt
    assert "规范化完成结果" in prompt
    assert "外部 Agent 已经执行任务后交付的完整结果" in prompt
    assert "完整性优先于简短" in prompt
    assert "不得用“已整理”“已标出”等概述" in prompt
    assert "列表项、名称、版本、数量、大小、路径、链接" in prompt
    assert "让用户再次索取详情的建议" in prompt
    for forbidden in (
        "private-endpoint",
        "credential:private-ref",
        "external-target-secret-id",
        "external-session-private-id",
        "external-run-private-id",
        "raw task input",
        "raw provider failure",
        "raw_tool_stream",
    ):
        assert forbidden not in prompt


@pytest.mark.asyncio
async def test_interaction_prompt_keeps_only_required_safe_provider_fields() -> None:
    client = FakeCompletionClient('{"m":"neutral","a":"要继续发布吗？"}')
    composer = AgentMessageComposer(client, FakeSoulLoader())

    draft = await composer.compose_interaction(interaction_work())

    assert draft.content == "要继续发布吗？"
    prompt = json.dumps(client.calls[0], ensure_ascii=False)
    assert "deploy safe-app" in prompt
    assert "发布已经审核的版本" in prompt
    assert "once" in prompt and "deny" in prompt
    assert "external-request-private-id" not in prompt
    assert "private tool output" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion",
    [
        '{"m":"neutral","m":"happy","a":"duplicate"}',
        '{"m":"neutral","a":"ok","extra":true}',
        '{"m":"excited","a":"unknown emotion"}',
        '{"m":"neutral","a":"   "}',
        '{"m":"neutral","a":"bad\\u0000text"}',
        '{"m":"neutral","a":"' + "好" * 4_001 + '"}',
        " " * 16_385,
    ],
)
async def test_invalid_or_oversized_completion_uses_fixed_neutral_fallback(
    completion: str,
) -> None:
    composer = AgentMessageComposer(FakeCompletionClient(completion), FakeSoulLoader())

    draft = await composer.compose_run(
        run_work(output="provider output that must never enter fallback")
    )

    assert draft.emotion == "neutral"
    assert draft.content == SAFE_AGENT_MESSAGE_FALLBACK
    assert draft.used_fallback is True
    assert "provider output" not in draft.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_value", "soul"),
    [
        (RuntimeError("Bearer private-token"), FakeSoulLoader()),
        ('{"m":"neutral","a":"unused"}', FakeSoulLoader(profile=None)),
        (
            '{"m":"neutral","a":"unused"}',
            FakeSoulLoader(error=RuntimeError("profile private detail")),
        ),
        (
            '{"m":"neutral","a":"unused"}',
            FakeSoulLoader(profile=SoulProfile("x" * 201, "setting")),
        ),
    ],
)
async def test_provider_and_profile_failures_materialize_same_safe_fallback(
    client_value: str | BaseException,
    soul: FakeSoulLoader,
) -> None:
    client = FakeCompletionClient(client_value)
    composer = AgentMessageComposer(client, soul)

    draft = await composer.compose_interaction(interaction_work())

    assert draft.emotion == "neutral"
    assert draft.content == SAFE_AGENT_MESSAGE_FALLBACK
    assert draft.used_fallback is True
    assert "private" not in draft.content


@pytest.mark.asyncio
async def test_invalid_bounded_work_input_also_uses_safe_fallback() -> None:
    client = FakeCompletionClient('{"m":"neutral","a":"must not run"}')
    composer = AgentMessageComposer(client, FakeSoulLoader())
    work = run_work()
    invalid = replace(work, task=replace(work.task, title=""))

    draft = await composer.compose_run(invalid)

    assert draft.content == SAFE_AGENT_MESSAGE_FALLBACK
    assert draft.used_fallback is True
    assert client.calls == []


class FakeGate:
    def __init__(self, active: bool = False) -> None:
        self.active = active
        self.calls = 0

    async def is_active(self) -> bool:
        self.calls += 1
        return self.active


class FakeDeliveryStore:
    def __init__(self) -> None:
        self.interaction: AgentInteractionWorkItem | None = None
        self.run: AgentRunWorkItem | None = None
        self.audit: list[str] = []
        self.materialized_interactions: list[tuple[UUID, str]] = []
        self.materialized_runs: list[tuple[UUID, str]] = []
        self.materialized_emotions: list[str] = []

    async def peek_unmaterialized_interaction(
        self,
    ) -> AgentInteractionWorkItem | None:
        self.audit.append("peek_interaction")
        return self.interaction

    async def peek_unmaterialized_terminal_run(self) -> AgentRunWorkItem | None:
        self.audit.append("peek_run")
        return self.run

    async def materialize_interaction_request(
        self,
        interaction_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> object | None:
        self.audit.append("materialize_interaction")
        self.materialized_interactions.append((interaction_id, content))
        self.materialized_emotions.append(emotion)
        return object()

    async def materialize_run_result(
        self,
        run_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> object | None:
        self.audit.append("materialize_run")
        self.materialized_runs.append((run_id, content))
        self.materialized_emotions.append(emotion)
        return object()


class FakeComposer:
    def __init__(self, emotion: str = "neutral") -> None:
        self.emotion = emotion
        self.audit: list[str] = []
        self.active = 0
        self.maximum_active = 0
        self.release = asyncio.Event()
        self.block = False

    async def compose_interaction(self, work: AgentInteractionWorkItem):
        self.audit.append("compose_interaction")
        return await self._compose("interaction")

    async def compose_run(self, work: AgentRunWorkItem):
        self.audit.append("compose_run")
        return await self._compose("run")

    async def _compose(self, label: str):
        from fish_memory.agent.message_delivery import ComposedAgentMessage

        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.block:
                await self.release.wait()
            return ComposedAgentMessage(self.emotion, f"{label} composed", False)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_processor_prioritizes_interaction_and_composes_outside_store_cas() -> (
    None
):
    store = FakeDeliveryStore()
    store.interaction = interaction_work()
    store.run = run_work()
    composer = FakeComposer()
    processor = AgentMessageProcessor(
        store,
        composer,
        background_semaphore=asyncio.Semaphore(1),
        foreground_gate=FakeGate(),
    )

    assert await processor.process_once() is True

    assert store.audit == ["peek_interaction", "materialize_interaction"]
    assert composer.audit == ["compose_interaction"]
    assert store.materialized_interactions == [(INTERACTION_ID, "interaction composed")]
    assert store.materialized_emotions == ["neutral"]
    assert store.materialized_runs == []


@pytest.mark.asyncio
async def test_processor_persists_composed_emotion_with_run_delivery() -> None:
    store = FakeDeliveryStore()
    store.run = run_work()
    processor = AgentMessageProcessor(
        store,
        FakeComposer("happy"),
        background_semaphore=asyncio.Semaphore(1),
        foreground_gate=FakeGate(),
    )

    assert await processor.process_once() is True
    assert store.materialized_emotions == ["happy"]


@pytest.mark.asyncio
async def test_active_foreground_gate_starts_no_selector_or_composer() -> None:
    store = FakeDeliveryStore()
    store.interaction = interaction_work()
    composer = FakeComposer()
    gate = FakeGate(active=True)
    processor = AgentMessageProcessor(
        store,
        composer,
        background_semaphore=asyncio.Semaphore(1),
        foreground_gate=gate,
    )

    assert await processor.process_once() is False
    assert store.audit == []
    assert composer.audit == []


@pytest.mark.asyncio
async def test_processors_share_only_injected_background_semaphore() -> None:
    first_store = FakeDeliveryStore()
    first_store.run = run_work()
    second_store = FakeDeliveryStore()
    second_store.run = replace(
        run_work(), run=replace(run_work().run, id=UUID(int=RUN_ID.int + 1))
    )
    composer = FakeComposer()
    composer.block = True
    semaphore = asyncio.Semaphore(1)
    first = AgentMessageProcessor(
        first_store,
        composer,
        background_semaphore=semaphore,
        foreground_gate=FakeGate(),
    )
    second = AgentMessageProcessor(
        second_store,
        composer,
        background_semaphore=semaphore,
        foreground_gate=FakeGate(),
    )

    tasks = [asyncio.create_task(item.process_once()) for item in (first, second)]
    for _ in range(100):
        if composer.active == 1:
            break
        await asyncio.sleep(0)
    assert composer.maximum_active == 1
    composer.release.set()
    assert await asyncio.gather(*tasks) == [True, True]
    assert composer.maximum_active == 1


@pytest.mark.asyncio
async def test_gate_is_rechecked_after_waiting_for_background_capacity() -> None:
    store = FakeDeliveryStore()
    store.run = run_work()
    composer = FakeComposer()
    gate = FakeGate()
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    processor = AgentMessageProcessor(
        store,
        composer,
        background_semaphore=semaphore,
        foreground_gate=gate,
    )

    task = asyncio.create_task(processor.process_once())
    for _ in range(100):
        if store.audit:
            break
        await asyncio.sleep(0)
    gate.active = True
    semaphore.release()

    assert await task is False
    assert store.audit == ["peek_interaction", "peek_run"]
    assert composer.audit == []
