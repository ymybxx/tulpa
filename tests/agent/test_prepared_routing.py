from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID

import pytest

from fish_memory.agent.models import AgentInteractionKind, AgentRun, AgentRunStatus
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyService,
    InputReplyAction,
    InputReplyClassification,
)
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteConflictError,
    AgentTaskRouteService,
    LocalConnectorCapabilityPolicy,
    TaskLaneContext,
)

from .test_replies import FakeClassifier, FakeStore, interaction_snapshot
from .test_routing import (
    CONFIRMATION_ID,
    FISH_ID,
    INTERACTION_ID,
    SESSION_ID,
    SOURCE_ID,
    TARGET_ID,
    RecordingTaskStore,
    candidate_for,
    resolved_target,
    run_for,
)


def context_for(
    status: AgentRunStatus, *, interaction: bool = False
) -> TaskLaneContext:
    return TaskLaneContext.from_records(
        (candidate_for(status, interaction=interaction),),
        current_content="重试这个任务",
        capability_policy=LocalConnectorCapabilityPolicy(),
    )


def input_store() -> FakeStore:
    snapshot = interaction_snapshot(AgentInteractionKind.INPUT)
    store = FakeStore(
        replace(snapshot, context=replace(snapshot.context, id=INTERACTION_ID))
    )
    store.handled_mutation = replace(
        store.handled_mutation, run=run_for(AgentRunStatus.QUEUED)
    )
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_mutation"),
    [
        (None, "new"),
        (AgentRunStatus.COMPLETED, "continue"),
        (AgentRunStatus.RUNNING, "steer"),
        (AgentRunStatus.UNKNOWN, "retry"),
    ],
)
async def test_prepare_does_not_queue_any_agent_work(
    status: AgentRunStatus | None, expected_mutation: str
) -> None:
    store = RecordingTaskStore()
    service = AgentTaskRouteService(store)
    indexes = [] if status is None else [0]
    context = TaskLaneContext.empty() if status is None else context_for(status)
    prepared = await service.prepare_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context,
        kind=(
            AgentTaskDecisionKind.NEW_TASK
            if status is None
            else AgentTaskDecisionKind.CONTINUE_TASK
        ),
        candidate_indexes=indexes,
        content="  重试这个任务  ",
    )

    assert store.calls == []
    assert prepared.content == "重试这个任务"
    assert prepared.target_id == TARGET_ID
    # The caller's candidate list is no longer consulted during commit.
    indexes[:] = [999]
    await service.commit_prepared_route(
        prepared,
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
    )
    assert [name for name, _ in store.calls] == [expected_mutation]


@pytest.mark.asyncio
async def test_disabling_tasks_rejects_a_prepared_chat_or_voice_commit():
    from fish_memory.agent.runtime_state import AgentRuntimeState

    state = AgentRuntimeState(enabled=True, default_target_id=TARGET_ID)
    store = RecordingTaskStore()
    service = AgentTaskRouteService(store, runtime_state=state)
    prepared = await service.prepare_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=TaskLaneContext.empty(),
        kind=AgentTaskDecisionKind.NEW_TASK,
        candidate_indexes=[],
        content="查阅资料",
    )
    state.configure(enabled=False, default_target_id=TARGET_ID)
    with pytest.raises(AgentTaskRouteConflictError, match="disabled"):
        await service.commit_prepared_route(
            prepared,
            source_message_id=SOURCE_ID,
            confirmation_message_id=CONFIRMATION_ID,
        )
    assert store.calls == []


@pytest.mark.asyncio
async def test_prepared_input_classification_runs_once_before_any_mutation() -> None:
    tasks = RecordingTaskStore()
    replies = input_store()
    classifier = FakeClassifier()
    service = AgentTaskRouteService(
        tasks, interaction_reply_service=AgentReplyService(replies, classifier)
    )
    prepared = await service.prepare_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context_for(AgentRunStatus.WAITING_USER, interaction=True),
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        content="Use release/next",
    )

    assert tasks.calls == replies.mutation_calls == []
    assert len(classifier.calls) == 1
    classifier.input_result = InputReplyClassification(InputReplyAction.UNRELATED)
    result = await service.commit_prepared_route(
        prepared,
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
    )

    assert result.status is AgentRunStatus.QUEUED
    assert len(classifier.calls) == 1
    assert tasks.calls == []
    assert [call[0] for call in replies.mutation_calls] == ["input"]
    assert replies.mutation_calls[0][2]["adopt_existing"] is True
    assert replies.mutation_calls[0][2]["message_id"] == SOURCE_ID


@pytest.mark.asyncio
async def test_unrelated_input_prepares_new_task_without_consuming_interaction() -> (
    None
):
    tasks = RecordingTaskStore()
    replies = input_store()
    classifier = FakeClassifier()
    classifier.input_result = InputReplyClassification(InputReplyAction.UNRELATED)
    service = AgentTaskRouteService(
        tasks, interaction_reply_service=AgentReplyService(replies, classifier)
    )
    prepared = await service.prepare_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context_for(AgentRunStatus.WAITING_USER, interaction=True),
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        content="Investigate a different issue",
    )

    assert prepared.kind is AgentTaskDecisionKind.NEW_TASK
    assert prepared.candidate is prepared.interaction_reply is None
    assert tasks.calls == replies.mutation_calls == []
    classifier.input_result = InputReplyClassification(InputReplyAction.ANSWER)
    await service.commit_prepared_route(
        prepared,
        source_message_id=SOURCE_ID,
        confirmation_message_id=CONFIRMATION_ID,
    )
    assert [name for name, _ in tasks.calls] == ["new"]
    assert replies.mutation_calls == []
    assert len(classifier.calls) == 1


@pytest.mark.asyncio
async def test_cancelling_input_preparation_never_queues_or_consumes_work() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClassifier(FakeClassifier):
        async def classify_input(
            self, *, request: str, content: str
        ) -> InputReplyClassification:
            entered.set()
            await release.wait()
            return await super().classify_input(request=request, content=content)

    tasks = RecordingTaskStore()
    replies = input_store()
    service = AgentTaskRouteService(
        tasks,
        interaction_reply_service=AgentReplyService(replies, BlockingClassifier()),
    )
    pending = asyncio.create_task(
        service.prepare_selected_route(
            fish_id=FISH_ID,
            resolved_target=resolved_target(),
            task_context=context_for(AgentRunStatus.WAITING_USER, interaction=True),
            kind=AgentTaskDecisionKind.CONTINUE_TASK,
            candidate_indexes=(0,),
            content="Use release/next",
        )
    )
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()
    await asyncio.sleep(0)
    assert tasks.calls == replies.mutation_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["source_content", "source_message_id"])
async def test_prepared_reply_rechecks_frozen_subject_without_reclassification(
    changed_field: str,
) -> None:
    store = input_store()
    classifier = FakeClassifier()
    service = AgentReplyService(store, classifier)
    prepared = await service.prepare_interaction_reply(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        interaction_id=INTERACTION_ID,
        content="Use release/next",
    )
    if changed_field == "source_content":
        store.snapshot = replace(store.snapshot, source_content="Another question")
    else:
        store.snapshot = replace(store.snapshot, source_message_id=UUID(int=999))

    with pytest.raises(AgentReplyConflictError, match="subject changed"):
        await service.commit_prepared_interaction(
            prepared,
            source_message_id=SOURCE_ID,
            confirmation_message_id=CONFIRMATION_ID,
        )
    assert len(classifier.calls) == 1
    assert store.mutation_calls == []


@pytest.mark.asyncio
async def test_prepared_route_still_requires_live_run_state_cas() -> None:
    class ChangedRunStore(RecordingTaskStore):
        async def queue_running_steer(
            self, run_id: UUID, source_message_id: UUID
        ) -> AgentRun | None:
            self.calls.append(("steer_cas_lost", (run_id, source_message_id)))
            return None

    store = ChangedRunStore()
    service = AgentTaskRouteService(store)
    prepared = await service.prepare_selected_route(
        fish_id=FISH_ID,
        resolved_target=resolved_target(),
        task_context=context_for(AgentRunStatus.RUNNING),
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(0,),
        content="Use Rust instead",
    )
    assert store.calls == []
    with pytest.raises(AgentTaskRouteConflictError, match="state CAS"):
        await service.commit_prepared_route(
            prepared,
            source_message_id=SOURCE_ID,
            confirmation_message_id=CONFIRMATION_ID,
        )
    assert [name for name, _ in store.calls] == ["steer_cas_lost"]


@pytest.mark.asyncio
async def test_prepared_interaction_does_not_bypass_store_cas_disposition() -> None:
    store = input_store()
    classifier = FakeClassifier()
    service = AgentReplyService(store, classifier)
    prepared = await service.prepare_interaction_reply(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        interaction_id=INTERACTION_ID,
        content="Use release/next",
    )
    store.handled_mutation = replace(store.handled_mutation, handled=False, run=None)
    with pytest.raises(AgentReplyConflictError, match="did not bind"):
        await service.commit_prepared_interaction(
            prepared,
            source_message_id=SOURCE_ID,
            confirmation_message_id=CONFIRMATION_ID,
        )
    assert len(classifier.calls) == 1
    assert [call[0] for call in store.mutation_calls] == ["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed", "recovered", "error"),
    [(("deny",), False, "not allowed"), (("once", "deny"), True, "only be denied")],
)
async def test_prepare_approval_keeps_allowed_and_recovered_policy(
    allowed: tuple[str, ...], recovered: bool, error: str
) -> None:
    snapshot = interaction_snapshot(
        AgentInteractionKind.APPROVAL,
        allowed_decisions=allowed,
        recovered_without_details=recovered,
    )
    store = FakeStore(snapshot)
    classifier = FakeClassifier()
    service = AgentReplyService(store, classifier)
    with pytest.raises(AgentReplyConflictError, match=error):
        await service.prepare_interaction_reply(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            interaction_id=snapshot.context.id,
            content="允许",
        )
    assert classifier.calls == store.mutation_calls == []
