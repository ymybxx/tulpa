import asyncio
from dataclasses import replace

import pytest

from fish_memory.agent.models import AgentInteractionKind
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyDisposition,
    AgentReplyReplay,
    AgentReplyService,
    InputReplyAction,
    InputReplyClassification,
    RunResultReplyAction,
    RunResultReplyClassification,
)

from .test_replies import (
    FISH_ID,
    MESSAGE_ID,
    NOW,
    SESSION_ID,
    FakeClassifier,
    FakeStore,
    interaction_snapshot,
    result_snapshot,
    user_message,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subject", "content", "action", "disposition", "requires_answer"),
    [
        ("approval", "允许", None, AgentReplyDisposition.APPROVAL, False),
        ("approval", "好吧", None, AgentReplyDisposition.ORDINARY, True),
        ("input", "main", InputReplyAction.ANSWER, AgentReplyDisposition.INPUT, False),
        (
            "input",
            "Other task",
            InputReplyAction.UNRELATED,
            AgentReplyDisposition.ORDINARY,
            True,
        ),
        (
            "result",
            "thanks",
            RunResultReplyAction.ACKNOWLEDGE,
            AgentReplyDisposition.ORDINARY,
            True,
        ),
        (
            "result",
            "Other task",
            RunResultReplyAction.UNRELATED,
            AgentReplyDisposition.ORDINARY,
            True,
        ),
        (
            "result",
            "continue",
            RunResultReplyAction.CONTINUE,
            AgentReplyDisposition.RESULT,
            False,
        ),
        (
            "result",
            "retry",
            RunResultReplyAction.RETRY,
            AgentReplyDisposition.RESULT,
            False,
        ),
        (
            "result",
            "change this",
            RunResultReplyAction.CHANGES_REQUESTED,
            AgentReplyDisposition.RESULT,
            False,
        ),
    ],
)
async def test_reply_plan_preserves_each_disposition_without_reclassification(
    subject: str,
    content: str,
    action: InputReplyAction | RunResultReplyAction | None,
    disposition: AgentReplyDisposition,
    requires_answer: bool,
) -> None:
    snapshot = (
        result_snapshot()
        if subject == "result"
        else interaction_snapshot(
            AgentInteractionKind(subject), allowed_decisions=("once", "deny")
        )
    )
    store = FakeStore(snapshot)
    classifier = FakeClassifier()
    if isinstance(action, InputReplyAction):
        classifier.input_result = InputReplyClassification(action)
    if isinstance(action, RunResultReplyAction):
        classifier.result_result = RunResultReplyClassification(action)
    service = AgentReplyService(store, classifier)
    prepared = await service.prepare_reply(
        FISH_ID, SESSION_ID, content, snapshot.context
    )
    assert prepared.disposition is disposition
    assert prepared.requires_answer is requires_answer
    assert prepared.handled is not requires_answer
    assert store.mutation_calls == []
    classification_count = len(classifier.calls)
    assert classification_count == int(subject != "approval")

    # A changed model disposition must not influence the prepared commit.
    classifier.input_result = InputReplyClassification(InputReplyAction.UNRELATED)
    classifier.result_result = RunResultReplyClassification(
        RunResultReplyAction.UNRELATED
    )
    result = await service.commit_prepared_reply(
        prepared, message_id=MESSAGE_ID, created_at=NOW
    )
    assert result.answer_expected is requires_answer
    assert len(classifier.calls) == classification_count
    assert [call[0] for call in store.mutation_calls] == [disposition.value]
    assert store.mutation_calls[0][2]["message_id"] == MESSAGE_ID
    assert store.mutation_calls[0][2]["created_at"] == NOW


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
async def test_commit_observes_durable_replay_created_after_preparation(
    handled: bool,
) -> None:
    snapshot = result_snapshot()
    store = FakeStore(snapshot)
    classifier = FakeClassifier()
    classifier.result_result = RunResultReplyClassification(
        RunResultReplyAction.CONTINUE
    )
    service = AgentReplyService(store, classifier)
    prepared = await service.prepare_reply(
        FISH_ID, SESSION_ID, "continue", snapshot.context
    )
    replay_message = user_message("continue")
    store.snapshot = replace(
        snapshot, replay=AgentReplyReplay(replay_message, handled=handled)
    )
    result = await service.commit_prepared_reply(prepared, message_id=MESSAGE_ID)
    assert result.replayed is True
    assert result.handled is handled
    assert result.message == replay_message
    assert store.mutation_calls == []
    assert len(classifier.calls) == 1


@pytest.mark.asyncio
async def test_changed_subject_cannot_consume_prepared_reply() -> None:
    snapshot = interaction_snapshot(AgentInteractionKind.INPUT)
    store = FakeStore(snapshot)
    classifier = FakeClassifier()
    service = AgentReplyService(store, classifier)
    prepared = await service.prepare_reply(
        FISH_ID, SESSION_ID, "main", snapshot.context
    )
    store.snapshot = replace(snapshot, source_content="A different question")
    with pytest.raises(AgentReplyConflictError, match="subject changed"):
        await service.commit_prepared_reply(prepared, message_id=MESSAGE_ID)
    assert store.mutation_calls == []
    assert len(classifier.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_reply_preparation_does_not_insert_or_dispatch() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClassifier(FakeClassifier):
        async def classify_input(
            self, *, request: str, content: str
        ) -> InputReplyClassification:
            entered.set()
            await release.wait()
            return InputReplyClassification(InputReplyAction.ANSWER)

    snapshot = interaction_snapshot(AgentInteractionKind.INPUT)
    store = FakeStore(snapshot)
    service = AgentReplyService(store, BlockingClassifier())
    pending = asyncio.create_task(
        service.prepare_reply(FISH_ID, SESSION_ID, "main", snapshot.context)
    )
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()
    assert store.mutation_calls == []
