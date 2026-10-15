from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionKind,
    AgentMessageSubjectKind,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentTarget,
    AgentTask,
    AgentTaskCandidate,
    ConnectorTypeSlug,
)
from fish_memory.agent.prompt import TASK_LANE_CONTRACT, build_task_lane_messages
from fish_memory.agent.replies import (
    AgentReplyContext,
    AgentReplySubjectSnapshot,
    PreparedAgentInteractionReply,
)
from fish_memory.agent.routing import (
    AgentInteractionReplyUnrelatedError,
    AgentTaskDecisionKind,
    AgentTaskRouteConflictError,
    AgentTaskRouteMode,
    AgentTaskRouteService,
    LocalConnectorCapabilityPolicy,
    TaskLaneContext,
)
from fish_memory.answer.speech_style import SPOKEN_TEXT_RULES

NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)
FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=8)
TARGET_ID = UUID(int=2)
TASK_ID = UUID(int=3)
RUN_ID = UUID(int=4)
INTERACTION_ID = UUID(int=5)
SOURCE_ID = UUID(int=6)
CONFIRMATION_ID = UUID(int=7)


def resolved_target(*, connector_type: str = "hermes") -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID(int=20),
        connector_type=ConnectorTypeSlug(connector_type),
        base_url="https://internal.example.invalid/api",
        created_at=NOW,
        updated_at=NOW,
    )
    target = AgentTarget(
        id=TARGET_ID,
        endpoint_id=endpoint.id,
        external_key="private-profile",
        route_prefix="/p/private",
        credential_ref="credential:must-not-leak",
        connector_config={"providerRaw": "must-not-leak"},
        display_name="Private target",
        enabled=True,
        created_at=NOW,
        updated_at=NOW,
    )
    return AgentResolvedTarget(target=target, endpoint=endpoint)


def run_for(status: AgentRunStatus, *, target_id: UUID = TARGET_ID) -> AgentRun:
    return AgentRun(
        id=RUN_ID,
        task_id=TASK_ID,
        target_id=target_id,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        previous_run_id=None,
        source_message_id=UUID(int=30),
        confirmation_message_id=UUID(int=31),
        input="original input",
        external_run_id="provider-run-must-not-leak",
        history_source=AgentHistorySource.NONE,
        status=status,
        output="provider output must not leak",
        result_message_id=UUID(int=32) if status.is_terminal else None,
        error_code=None,
        error_message=None,
        provider_data={"raw": "provider-data-must-not-leak"},
        created_at=NOW,
        started_at=NOW,
        updated_at=NOW,
        finished_at=NOW if status.is_terminal else None,
    )


def candidate_for(
    status: AgentRunStatus,
    *,
    connector_type: str = "hermes",
    interaction: bool = False,
) -> AgentTaskCandidate:
    resolved = resolved_target(connector_type=connector_type)
    task = AgentTask(
        id=TASK_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        target_id=resolved.target.id,
        external_session_id="provider-session-must-not-leak",
        title="修复测试任务",
        created_at=NOW,
        updated_at=NOW,
    )
    open_interaction = (
        AgentInteraction(
            id=INTERACTION_ID,
            run_id=RUN_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            kind=AgentInteractionKind.INPUT,
            request_message_id=UUID(int=40),
            response_message_id=None,
            next_run_id=None,
            provider_data={"raw": "interaction-provider-data-must-not-leak"},
            created_at=NOW,
            resolved_at=None,
            closed_at=None,
        )
        if interaction
        else None
    )
    return AgentTaskCandidate(
        task=task,
        run=run_for(status),
        interaction=open_interaction,
        target=resolved.target,
        endpoint=resolved.endpoint,
        interaction_request_content=(
            "多久后重新检查内存？" if open_interaction is not None else None
        ),
        interaction_request_session_id=(
            SESSION_ID if open_interaction is not None else None
        ),
    )


@pytest.mark.parametrize(
    ("status", "content", "connector_type", "interaction", "expected"),
    [
        (AgentRunStatus.COMPLETED, "继续这个任务", "hermes", False, "new_run"),
        (AgentRunStatus.RUNNING, "把实现改成 Rust", "hermes", False, "steer"),
        (AgentRunStatus.UNKNOWN, "重试这个任务", "hermes", False, "retry"),
        (AgentRunStatus.QUEUED, "继续这个任务", "hermes", False, "reply_only"),
        (AgentRunStatus.DISPATCHING, "继续这个任务", "hermes", False, "reply_only"),
        (AgentRunStatus.WAITING_USER, "继续这个任务", "hermes", True, "reply_only"),
        (AgentRunStatus.RUNNING, "继续这个任务", "hermes", False, "steer"),
        (AgentRunStatus.RUNNING, "把实现改成 Rust", "future", False, "reply_only"),
    ],
)
def test_task_lane_context_derives_route_mode_only_from_local_state_and_policy(
    status: AgentRunStatus,
    content: str,
    connector_type: str,
    interaction: bool,
    expected: str,
) -> None:
    context = TaskLaneContext.from_records(
        (
            candidate_for(
                status, connector_type=connector_type, interaction=interaction
            ),
        ),
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    assert context.candidates[0].index == 0
    assert context.candidates[0].mode is AgentTaskRouteMode(expected)


def test_task_prompt_exposes_stable_indexes_but_no_internal_or_provider_fields() -> (
    None
):
    record = candidate_for(AgentRunStatus.RUNNING, interaction=True)
    context = TaskLaneContext.from_records(
        (record,),
        current_content="继续修复测试任务",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    messages = build_task_lane_messages(
        [
            {"role": "system", "content": "fish soul"},
            {"role": "user", "content": "继续修复测试任务"},
        ],
        context,
    )
    payload = "\n".join(message["content"] for message in messages)

    assert '"i":0' in payload
    assert "original input" in payload
    assert '"request":"多久后重新检查内存？"' in payload
    assert context.prompt_payload() == [
        {
            "i": 0,
            "title": "original input",
            "request": "多久后重新检查内存？",
        }
    ]
    for forbidden in (
        str(record.task.id),
        str(record.run.id),
        str(record.interaction.id) if record.interaction is not None else "",
        record.endpoint.base_url,
        cast(str, record.target.credential_ref),
        cast(str, record.run.external_run_id),
        "provider-data-must-not-leak",
        "interaction-provider-data-must-not-leak",
        record.task.external_session_id,
    ):
        assert forbidden and forbidden not in payload

    assert '"k"' not in payload
    assert '"t":2' not in payload
    assert "fish soul" not in payload
    assert "无关联或候选为空时省略 c，表示新任务" in payload
    assert messages[0] == {"role": "system", "content": TASK_LANE_CONTRACT}
    current_payload = messages[-1]["content"]
    assert current_payload.startswith("# 当前用户消息与候选\n")
    decoded = json.loads(current_payload.split("\n", 1)[1])
    assert list(decoded) == ["candidates", "content"]
    assert decoded["content"] == "继续修复测试任务"


def test_task_prompt_is_a_compact_two_way_capability_router() -> None:
    assert "t=1：目标超出宿主当前能力与活动职责" in TASK_LANE_CONTRACT
    assert "t=0：普通交流、利用已提供的感知回答" in TASK_LANE_CONTRACT
    assert "单纯询问宿主的能力、感知或权限状态属于普通交流" in TASK_LANE_CONTRACT
    assert "不得将这种询问扩写成权限诊断、能力验证或工具调用任务" in TASK_LANE_CONTRACT
    assert "先冻结 t" in TASK_LANE_CONTRACT
    assert "候选存在、唯一、最近、未完成或相邻 Agent 回复都不能改变 t" in (
        TASK_LANE_CONTRACT
    )
    assert "t=0 时 a 必须为空" in TASK_LANE_CONTRACT
    assert "a 的语言只由当前 content 决定" in TASK_LANE_CONTRACT
    assert "确认当前具体" in TASK_LANE_CONTRACT
    assert "无目标信息的通用套话" in TASK_LANE_CONTRACT
    assert "不能提问或声称" in TASK_LANE_CONTRACT
    assert "任务已完成" in TASK_LANE_CONTRACT
    assert TASK_LANE_CONTRACT.endswith(SPOKEN_TEXT_RULES)
    assert len(TASK_LANE_CONTRACT.removesuffix(SPOKEN_TEXT_RULES)) < 800
    assert "t=2" not in TASK_LANE_CONTRACT
    assert "Interaction" not in TASK_LANE_CONTRACT
    assert "Provider" not in TASK_LANE_CONTRACT


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (None, AgentTaskDecisionKind.NEW_TASK),
        (0, AgentTaskDecisionKind.CONTINUE_TASK),
    ],
)
def test_task_route_kind_is_derived_from_compact_candidate_index(
    index: int | None,
    expected: AgentTaskDecisionKind,
) -> None:
    assert AgentTaskDecisionKind.from_candidate_index(index) is expected
    assert list(AgentTaskDecisionKind) == [
        AgentTaskDecisionKind.NEW_TASK,
        AgentTaskDecisionKind.CONTINUE_TASK,
    ]


def test_task_route_kind_rejects_invalid_compact_candidate_index() -> None:
    with pytest.raises(ValueError, match="integer"):
        AgentTaskDecisionKind.from_candidate_index(True)


def test_exact_fingerprint_has_internal_mapping_and_policy() -> None:
    context = TaskLaneContext.from_records(
        (candidate_for(AgentRunStatus.UNKNOWN),),
        current_content="重试这个任务",
        capability_policy=LocalConnectorCapabilityPolicy(
            routing_policy_version="test-policy-v9"
        ),
    )

    assert context.fingerprint_payload() == {
        "routingPolicyVersion": "test-policy-v9",
        "hasRecentAgentReply": False,
        "candidates": [
            {
                "index": 0,
                "taskId": str(TASK_ID),
                "runId": str(RUN_ID),
                "interactionId": None,
                "targetId": str(TARGET_ID),
                "mode": "retry",
            }
        ],
    }


def test_task_context_requires_probe_for_recent_reply_or_unfinished_task() -> None:
    completed = candidate_for(AgentRunStatus.COMPLETED)
    recent = TaskLaneContext.from_records(
        (completed,),
        current_content="还要好吃的",
        capability_policy=LocalConnectorCapabilityPolicy(),
        has_recent_agent_reply=True,
    )
    stale = TaskLaneContext.from_records(
        (completed,),
        current_content="还要好吃的",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )
    running = TaskLaneContext.from_records(
        (candidate_for(AgentRunStatus.RUNNING),),
        current_content="还要好吃的",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    assert recent.has_recent_agent_reply is True
    assert recent.requires_probe is True
    assert stale.requires_probe is False
    assert running.has_unfinished_task is True
    assert running.requires_probe is True
    assert running.candidates[0].mode is AgentTaskRouteMode.STEER


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.QUEUED,
        AgentRunStatus.DISPATCHING,
        AgentRunStatus.RUNNING,
        AgentRunStatus.WAITING_USER,
        AgentRunStatus.UNKNOWN,
    ],
)
def test_every_unfinished_run_status_requires_t_probe(status: AgentRunStatus) -> None:
    context = TaskLaneContext.from_records(
        (candidate_for(status),),
        current_content="还要好吃的",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    assert context.has_unfinished_task is True
    assert context.requires_probe is True


def test_capability_policy_tracks_supported_connectors_and_steer_subset() -> None:
    policy = LocalConnectorCapabilityPolicy(
        supported_connector_types=frozenset({"hermes", "fake.valid"}),
        steer_connector_types=frozenset({"hermes"}),
    )

    assert policy.supports_connector("hermes") is True
    assert policy.supports_connector("fake.valid") is True
    assert policy.supports_connector("missing") is False
    assert policy.supports_steer("hermes") is True
    assert policy.supports_steer("fake.valid") is False


def test_capability_policy_rejects_steer_connector_outside_supported_set() -> None:
    with pytest.raises(ValueError, match="steer connector"):
        LocalConnectorCapabilityPolicy(
            supported_connector_types=frozenset({"hermes"}),
            steer_connector_types=frozenset({"fake.valid"}),
        )


def test_interaction_identity_changes_exact_fingerprint_without_leaking_to_prompt() -> (
    None
):
    first_record = candidate_for(AgentRunStatus.WAITING_USER, interaction=True)
    assert first_record.interaction is not None
    second_record = replace(
        first_record,
        interaction=replace(first_record.interaction, id=UUID(int=505)),
    )
    policy = LocalConnectorCapabilityPolicy()
    first = TaskLaneContext.from_records(
        (first_record,),
        current_content="继续这个任务",
        capability_policy=policy,
    )
    second = TaskLaneContext.from_records(
        (second_record,),
        current_content="继续这个任务",
        capability_policy=policy,
    )

    assert first.prompt_payload() == second.prompt_payload()
    assert first.fingerprint_payload() != second.fingerprint_payload()


class RecordingTaskStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def create_task_with_first_run(
        self,
        fish_id: UUID,
        target_id: UUID,
        title: str,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> tuple[AgentTask, AgentRun]:
        self.calls.append(
            (
                "new",
                (
                    fish_id,
                    target_id,
                    title,
                    source_message_id,
                    confirmation_message_id,
                    input,
                ),
            )
        )
        record = candidate_for(AgentRunStatus.QUEUED)
        return record.task, record.run

    async def create_continuation_run(
        self,
        task_id: UUID,
        expected_previous_run_id: UUID,
        source_message_id: UUID,
        input: str,
        confirmation_message_id: UUID | None,
    ) -> AgentRun:
        self.calls.append(
            (
                "continue",
                (
                    task_id,
                    expected_previous_run_id,
                    source_message_id,
                    input,
                    confirmation_message_id,
                ),
            )
        )
        return run_for(AgentRunStatus.QUEUED)

    async def queue_running_steer(
        self,
        run_id: UUID,
        source_message_id: UUID,
    ) -> AgentRun | None:
        self.calls.append(("steer", (run_id, source_message_id)))
        return run_for(AgentRunStatus.RUNNING)

    async def abandon_unknown_and_create_retry(
        self,
        run_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> AgentRun | None:
        self.calls.append(
            (
                "retry",
                (run_id, source_message_id, confirmation_message_id, input),
            )
        )
        return run_for(AgentRunStatus.QUEUED)


class RecordingInteractionReplyService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.prepare_calls: list[dict[str, object]] = []
        self.unrelated = False

    async def prepare_interaction_reply(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        interaction_id: UUID,
        content: str,
    ) -> PreparedAgentInteractionReply:
        self.prepare_calls.append(
            dict(
                fish_id=fish_id,
                session_id=session_id,
                interaction_id=interaction_id,
                content=content,
            )
        )
        if self.unrelated:
            raise AgentInteractionReplyUnrelatedError("unrelated")
        return PreparedAgentInteractionReply(
            fish_id,
            session_id,
            content,
            AgentReplySubjectSnapshot(
                context=AgentReplyContext(
                    AgentMessageSubjectKind.INTERACTION, interaction_id
                ),
                source_message_id=UUID(int=40),
                source_content="多久后重新检查内存？",
                interaction_kind=AgentInteractionKind.INPUT,
                allowed_decisions=(),
                recovered_without_details=False,
                replay=None,
            ),
        )

    async def commit_prepared_interaction(
        self,
        prepared: PreparedAgentInteractionReply,
        *,
        source_message_id: UUID,
        confirmation_message_id: UUID,
    ) -> AgentRun | None:
        self.calls.append(
            dict(
                fish_id=prepared.fish_id,
                session_id=prepared.session_id,
                interaction_id=prepared.snapshot.context.id,
                content=prepared.content,
                source_message_id=source_message_id,
                confirmation_message_id=confirmation_message_id,
            )
        )
        return run_for(AgentRunStatus.QUEUED)


@pytest.mark.asyncio
async def test_new_task_binds_the_current_confirmation_message_to_the_first_run() -> (
    None
):
    store = RecordingTaskStore()
    service = AgentTaskRouteService(store)
    target = resolved_target()

    await service.commit_selected_route(
        fish_id=FISH_ID,
        resolved_target=target,
        task_context=TaskLaneContext.empty(),
        kind=AgentTaskDecisionKind.NEW_TASK,
        candidate_indexes=(),
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
        content="调查故障并给我报告",
    )

    assert store.calls == [
        (
            "new",
            (
                FISH_ID,
                target.target.id,
                "调查故障并给我报告",
                SOURCE_ID,
                CONFIRMATION_ID,
                "调查故障并给我报告",
            ),
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "content", "expected_call", "confirmation_position"),
    [
        (AgentRunStatus.COMPLETED, "继续这个任务", "continue", 4),
        (AgentRunStatus.RUNNING, "还要好吃的", "steer", None),
        (AgentRunStatus.UNKNOWN, "重试这个任务", "retry", 2),
    ],
)
async def test_continue_route_uses_the_locally_frozen_mode_and_candidate_target(
    status: AgentRunStatus,
    content: str,
    expected_call: str,
    confirmation_position: int | None,
) -> None:
    store = RecordingTaskStore()
    context = TaskLaneContext.from_records(
        (candidate_for(status),),
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    await AgentTaskRouteService(store).commit_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(connector_type="future"),
        task_context=context,
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
        content=content,
    )

    assert len(store.calls) == 1
    name, arguments = store.calls[0]
    assert name == expected_call
    if name == "continue":
        assert arguments[1] == RUN_ID
    if confirmation_position is None:
        assert CONFIRMATION_ID not in arguments
    else:
        assert arguments[confirmation_position] == CONFIRMATION_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "content"),
    [
        (AgentRunStatus.QUEUED, "继续这个任务"),
        (AgentRunStatus.DISPATCHING, "继续这个任务"),
    ],
)
async def test_reply_only_continue_fails_closed_without_a_store_write(
    status: AgentRunStatus,
    content: str,
) -> None:
    store = RecordingTaskStore()
    context = TaskLaneContext.from_records(
        (candidate_for(status),),
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    with pytest.raises(AgentTaskRouteConflictError, match="reply_only"):
        await AgentTaskRouteService(store).commit_selected_route(
            fish_id=FISH_ID,
            resolved_target=resolved_target(),
            task_context=context,
            kind=AgentTaskDecisionKind.CONTINUE_TASK,
            candidate_indexes=(0,),
            source_message_id=SOURCE_ID,
            confirmation_message_id=CONFIRMATION_ID,
            content=content,
        )

    assert store.calls == []


@pytest.mark.asyncio
async def test_continue_task_routes_waiting_interaction_to_reply_service() -> None:
    store = RecordingTaskStore()
    replies = RecordingInteractionReplyService()
    context = TaskLaneContext.from_records(
        (candidate_for(AgentRunStatus.WAITING_USER, interaction=True),),
        current_content="立刻",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    result = await AgentTaskRouteService(
        store,
        interaction_reply_service=replies,
    ).commit_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context,
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
        content="立刻",
    )

    assert result.status is AgentRunStatus.QUEUED
    assert store.calls == []
    assert replies.calls == [
        {
            "fish_id": FISH_ID,
            "session_id": SESSION_ID,
            "interaction_id": INTERACTION_ID,
            "source_message_id": SOURCE_ID,
            "confirmation_message_id": CONFIRMATION_ID,
            "content": "立刻",
        }
    ]


@pytest.mark.asyncio
async def test_unrelated_interaction_selection_falls_back_to_a_new_task() -> None:
    store = RecordingTaskStore()
    replies = RecordingInteractionReplyService()
    replies.unrelated = True
    context = TaskLaneContext.from_records(
        (candidate_for(AgentRunStatus.WAITING_USER, interaction=True),),
        current_content="看下斗罗大陆动漫完结了吗",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )

    result = await AgentTaskRouteService(
        store,
        interaction_reply_service=replies,
    ).commit_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context,
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
        content="看下斗罗大陆动漫完结了吗",
    )

    assert result.status is AgentRunStatus.QUEUED
    assert store.calls == [
        (
            "new",
            (
                FISH_ID,
                TARGET_ID,
                "看下斗罗大陆动漫完结了吗",
                SOURCE_ID,
                CONFIRMATION_ID,
                "看下斗罗大陆动漫完结了吗",
            ),
        )
    ]
