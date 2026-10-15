from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.agent.models import AgentInteractionKind, AgentMessageSubjectKind
from fish_memory.agent.replies import (
    AgentReplyClassificationError,
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyMutation,
    AgentReplyReplay,
    AgentReplyService,
    AgentReplySubjectSnapshot,
    InputReplyAction,
    InputReplyClassification,
    LlmReplyClassifier,
    RunResultReplyAction,
    RunResultReplyClassification,
)
from fish_memory.agent.routing import AgentInteractionReplyUnrelatedError
from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.models import JsonValue

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
SUBJECT_ID = UUID(int=3)
MESSAGE_ID = UUID(int=4)
NOW = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)


def user_message(content: str = "reply") -> Message:
    return Message(
        id=MESSAGE_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=11,
        role=Role.USER,
        content=content,
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )


def interaction_snapshot(
    kind: AgentInteractionKind,
    *,
    allowed_decisions: tuple[str, ...] = (),
    recovered_without_details: bool = False,
) -> AgentReplySubjectSnapshot:
    return AgentReplySubjectSnapshot(
        context=AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
        source_message_id=UUID(int=30),
        source_content="Which branch should I use?",
        interaction_kind=kind,
        allowed_decisions=allowed_decisions,
        recovered_without_details=recovered_without_details,
        replay=None,
    )


def result_snapshot() -> AgentReplySubjectSnapshot:
    return AgentReplySubjectSnapshot(
        context=AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, SUBJECT_ID),
        source_message_id=UUID(int=31),
        source_content="The run completed successfully.",
        interaction_kind=None,
        allowed_decisions=(),
        recovered_without_details=False,
        replay=None,
    )


class FakeStore:
    def __init__(self, snapshot: AgentReplySubjectSnapshot) -> None:
        self.snapshot = snapshot
        self.load_calls: list[tuple[object, ...]] = []
        self.mutation_calls: list[
            tuple[str, tuple[object, ...], dict[str, object]]
        ] = []
        self.handled_mutation = AgentReplyMutation(
            user_message(), inserted=True, replayed=False, handled=True
        )
        self.ordinary_mutation = AgentReplyMutation(
            user_message(), inserted=True, replayed=False, handled=False
        )

    async def load_reply_subject(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        *,
        message_id: UUID | None = None,
        content: str | None = None,
    ) -> AgentReplySubjectSnapshot:
        self.load_calls.append((context, fish_id, session_id, message_id, content))
        return self.snapshot

    async def insert_and_bind_approval_response(
        self,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        decision: object,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
        adopt_existing: bool = False,
    ) -> AgentReplyMutation:
        details: dict[str, object] = {
            "message_id": message_id,
            "created_at": created_at,
        }
        if adopt_existing:
            details["adopt_existing"] = True
        self.mutation_calls.append(
            (
                "approval",
                (interaction_id, fish_id, session_id, content, decision),
                details,
            )
        )
        return self.handled_mutation

    async def insert_and_resolve_input(
        self,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
        confirmation_message_id: UUID | None = None,
        adopt_existing: bool = False,
    ) -> AgentReplyMutation:
        details: dict[str, object] = {
            "message_id": message_id,
            "created_at": created_at,
        }
        if confirmation_message_id is not None:
            details["confirmation_message_id"] = confirmation_message_id
        if adopt_existing:
            details["adopt_existing"] = True
        self.mutation_calls.append(
            (
                "input",
                (interaction_id, fish_id, session_id, content),
                details,
            )
        )
        return self.handled_mutation

    async def insert_and_create_result_followup(
        self,
        run_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        action: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyMutation:
        self.mutation_calls.append(
            (
                "result",
                (run_id, fish_id, session_id, content, action),
                {"message_id": message_id, "created_at": created_at},
            )
        )
        return self.handled_mutation

    async def insert_validated_ordinary_reply(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyMutation:
        self.mutation_calls.append(
            (
                "ordinary",
                (context, fish_id, session_id, content),
                {"message_id": message_id, "created_at": created_at},
            )
        )
        return self.ordinary_mutation


class FakeClassifier:
    def __init__(self) -> None:
        self.input_result = InputReplyClassification(InputReplyAction.ANSWER)
        self.result_result = RunResultReplyClassification(
            RunResultReplyAction.ACKNOWLEDGE
        )
        self.calls: list[tuple[str, str, str]] = []

    async def classify_input(
        self, *, request: str, content: str
    ) -> InputReplyClassification:
        self.calls.append(("input", request, content))
        return self.input_result

    async def classify_run_result(
        self, *, result: str, content: str
    ) -> RunResultReplyClassification:
        self.calls.append(("result", result, content))
        return self.result_result


class FakeJsonClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[Sequence[ChatMessage]] = []
        self.error: Exception | None = None

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return JsonCompletion(self.responses.pop(0))

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return list(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["允许", "approve once", "yes, approve"])
async def test_explicit_approval_defaults_to_once_without_resolve_all(
    content: str,
) -> None:
    store = FakeStore(
        interaction_snapshot(
            AgentInteractionKind.APPROVAL,
            allowed_decisions=("once", "deny"),
        )
    )
    classifier = FakeClassifier()

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        content,
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
        message_id=MESSAGE_ID,
        created_at=NOW,
    )

    assert result.handled is True
    assert result.answer_expected is False
    assert classifier.calls == []
    assert store.mutation_calls == [
        (
            "approval",
            (
                SUBJECT_ID,
                FISH_ID,
                SESSION_ID,
                content,
                {"choice": "once", "resolve_all": False},
            ),
            {"message_id": MESSAGE_ID, "created_at": NOW},
        )
    ]


@pytest.mark.asyncio
async def test_explicit_broader_approval_requires_allowed_choice() -> None:
    store = FakeStore(
        interaction_snapshot(
            AgentInteractionKind.APPROVAL,
            allowed_decisions=("session", "always", "deny"),
        )
    )
    service = AgentReplyService(store, FakeClassifier())

    session = await service.handle(
        FISH_ID,
        SESSION_ID,
        "allow for this session",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )
    always = await service.handle(
        FISH_ID,
        SESSION_ID,
        "always allow",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )

    assert session.handled is always.handled is True
    assert store.mutation_calls[0][1][-1] == {
        "choice": "session",
        "resolve_all": False,
    }
    assert store.mutation_calls[1][1][-1] == {
        "choice": "always",
        "resolve_all": False,
    }

    store.snapshot = interaction_snapshot(
        AgentInteractionKind.APPROVAL,
        allowed_decisions=("once", "deny"),
    )
    all_pending = await service.handle(
        FISH_ID,
        SESSION_ID,
        "approve all pending requests",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )
    assert all_pending.handled is True
    assert store.mutation_calls[2][1][-1] == {
        "choice": "once",
        "resolve_all": True,
    }

    store.snapshot = interaction_snapshot(
        AgentInteractionKind.APPROVAL,
        allowed_decisions=("once", "deny"),
    )
    with pytest.raises(AgentReplyConflictError, match="not allowed"):
        await service.handle(
            FISH_ID,
            SESSION_ID,
            "always allow",
            AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
        )


@pytest.mark.asyncio
async def test_recovered_approval_accepts_only_explicit_deny() -> None:
    store = FakeStore(
        interaction_snapshot(
            AgentInteractionKind.APPROVAL,
            allowed_decisions=("deny",),
            recovered_without_details=True,
        )
    )
    service = AgentReplyService(store, FakeClassifier())

    denied = await service.handle(
        FISH_ID,
        SESSION_ID,
        "拒绝",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )

    assert denied.handled is True
    assert store.mutation_calls[0][1][-1] == {
        "choice": "deny",
        "resolve_all": False,
    }

    with pytest.raises(AgentReplyConflictError, match="only be denied"):
        await service.handle(
            FISH_ID,
            SESSION_ID,
            "允许",
            AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
        )


@pytest.mark.asyncio
async def test_ambiguous_approval_is_not_consumed_as_authorization() -> None:
    store = FakeStore(
        interaction_snapshot(
            AgentInteractionKind.APPROVAL,
            allowed_decisions=("once", "deny"),
        )
    )

    result = await AgentReplyService(store, FakeClassifier()).handle(
        FISH_ID,
        SESSION_ID,
        "好吧",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )

    assert result.handled is False
    assert result.answer_expected is True
    assert store.mutation_calls[0][0] == "ordinary"


@pytest.mark.asyncio
async def test_input_answer_is_classified_then_atomically_consumed() -> None:
    store = FakeStore(interaction_snapshot(AgentInteractionKind.INPUT))
    classifier = FakeClassifier()

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        "Use release/next",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
    )

    assert result.handled is True
    assert classifier.calls == [
        ("input", "Which branch should I use?", "Use release/next")
    ]
    assert store.mutation_calls[0][0] == "input"


@pytest.mark.asyncio
async def test_t_continue_classifies_then_adopts_existing_input() -> None:
    store = FakeStore(interaction_snapshot(AgentInteractionKind.INPUT))
    classifier = FakeClassifier()
    confirmation_id = UUID(int=40)

    result = await AgentReplyService(store, classifier).continue_interaction(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        interaction_id=SUBJECT_ID,
        source_message_id=MESSAGE_ID,
        confirmation_message_id=confirmation_id,
        content="Immediately",
    )

    assert result is None
    assert classifier.calls == [("input", "Which branch should I use?", "Immediately")]
    assert (
        store.load_calls
        == [
            (
                AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
                FISH_ID,
                SESSION_ID,
                None,
                None,
            )
        ]
        * 2
    )
    assert store.mutation_calls == [
        (
            "input",
            (SUBJECT_ID, FISH_ID, SESSION_ID, "Immediately"),
            {
                "message_id": MESSAGE_ID,
                "created_at": None,
                "confirmation_message_id": confirmation_id,
                "adopt_existing": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_t_continue_rejects_unrelated_input_without_consuming() -> None:
    store = FakeStore(interaction_snapshot(AgentInteractionKind.INPUT))
    classifier = FakeClassifier()
    classifier.input_result = InputReplyClassification(InputReplyAction.UNRELATED)

    with pytest.raises(AgentInteractionReplyUnrelatedError):
        await AgentReplyService(store, classifier).continue_interaction(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            interaction_id=SUBJECT_ID,
            source_message_id=MESSAGE_ID,
            confirmation_message_id=UUID(int=40),
            content="看下斗罗大陆动漫完结了吗",
        )

    assert classifier.calls == [
        (
            "input",
            "Which branch should I use?",
            "看下斗罗大陆动漫完结了吗",
        )
    ]
    assert store.mutation_calls == []


@pytest.mark.asyncio
async def test_t_continue_keeps_approval_parsing_fail_closed() -> None:
    store = FakeStore(
        interaction_snapshot(
            AgentInteractionKind.APPROVAL,
            allowed_decisions=("once", "deny"),
        )
    )
    service = AgentReplyService(store, FakeClassifier())

    await service.continue_interaction(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        interaction_id=SUBJECT_ID,
        source_message_id=MESSAGE_ID,
        confirmation_message_id=UUID(int=41),
        content="允许",
    )

    assert store.mutation_calls[0] == (
        "approval",
        (
            SUBJECT_ID,
            FISH_ID,
            SESSION_ID,
            "允许",
            {"choice": "once", "resolve_all": False},
        ),
        {
            "message_id": MESSAGE_ID,
            "created_at": None,
            "adopt_existing": True,
        },
    )

    store.mutation_calls.clear()
    with pytest.raises(AgentReplyConflictError, match="does not contain a decision"):
        await service.continue_interaction(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            interaction_id=SUBJECT_ID,
            source_message_id=MESSAGE_ID,
            confirmation_message_id=UUID(int=41),
            content="好吧",
        )
    assert store.mutation_calls == []


@pytest.mark.asyncio
async def test_final_store_mutation_overrides_input_classifier_disposition() -> None:
    store = FakeStore(interaction_snapshot(AgentInteractionKind.INPUT))
    store.handled_mutation = AgentReplyMutation(
        user_message(),
        inserted=False,
        replayed=True,
        handled=False,
    )

    result = await AgentReplyService(store, FakeClassifier()).handle(
        FISH_ID,
        SESSION_ID,
        "Use release/next",
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, SUBJECT_ID),
        message_id=MESSAGE_ID,
    )

    assert result.handled is False
    assert result.answer_expected is True


@pytest.mark.asyncio
async def test_final_store_mutation_overrides_unrelated_classifier_disposition() -> (
    None
):
    store = FakeStore(result_snapshot())
    store.ordinary_mutation = AgentReplyMutation(
        user_message(),
        inserted=False,
        replayed=True,
        handled=True,
    )
    classifier = FakeClassifier()
    classifier.result_result = RunResultReplyClassification(
        RunResultReplyAction.UNRELATED
    )

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        "continue",
        AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, SUBJECT_ID),
        message_id=MESSAGE_ID,
    )

    assert result.handled is True
    assert result.answer_expected is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "handled", "store_action"),
    [
        (RunResultReplyAction.ACKNOWLEDGE, False, "ordinary"),
        (RunResultReplyAction.UNRELATED, False, "ordinary"),
        (RunResultReplyAction.CONTINUE, True, "result"),
        (RunResultReplyAction.RETRY, True, "result"),
        (RunResultReplyAction.CHANGES_REQUESTED, True, "result"),
    ],
)
async def test_run_result_actions_have_fixed_answer_semantics(
    action: RunResultReplyAction,
    handled: bool,
    store_action: str,
) -> None:
    store = FakeStore(result_snapshot())
    classifier = FakeClassifier()
    classifier.result_result = RunResultReplyClassification(action)

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        "reply",
        AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, SUBJECT_ID),
    )

    assert result.handled is handled
    assert result.answer_expected is (not handled)
    assert store.mutation_calls[0][0] == store_action
    if store_action == "result":
        assert store.mutation_calls[0][1][-1] == action.value


@pytest.mark.asyncio
async def test_classifier_await_cannot_replace_frozen_subject_identity() -> None:
    original = result_snapshot()
    store = FakeStore(original)
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClassifier(FakeClassifier):
        async def classify_run_result(
            self, *, result: str, content: str
        ) -> RunResultReplyClassification:
            entered.set()
            await release.wait()
            return RunResultReplyClassification(RunResultReplyAction.CONTINUE)

    service = AgentReplyService(store, BlockingClassifier())
    pending = asyncio.create_task(
        service.handle(
            FISH_ID,
            SESSION_ID,
            "continue",
            AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, SUBJECT_ID),
        )
    )
    await entered.wait()
    store.snapshot = AgentReplySubjectSnapshot(
        context=AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=999)),
        source_message_id=UUID(int=998),
        source_content="newer result",
        interaction_kind=None,
        allowed_decisions=(),
        recovered_without_details=False,
        replay=None,
    )
    release.set()

    with pytest.raises(AgentReplyConflictError, match="subject identity changed"):
        await pending
    assert store.mutation_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        interaction_snapshot(AgentInteractionKind.APPROVAL),
        interaction_snapshot(AgentInteractionKind.INPUT),
        result_snapshot(),
    ],
)
async def test_durable_handled_replay_short_circuits_classifier_and_mutation(
    snapshot: AgentReplySubjectSnapshot,
) -> None:
    replay_message = user_message("same reply")
    replay_snapshot = AgentReplySubjectSnapshot(
        context=snapshot.context,
        source_message_id=snapshot.source_message_id,
        source_content=snapshot.source_content,
        interaction_kind=snapshot.interaction_kind,
        allowed_decisions=snapshot.allowed_decisions,
        recovered_without_details=snapshot.recovered_without_details,
        replay=AgentReplyReplay(message=replay_message, handled=True),
    )
    store = FakeStore(replay_snapshot)
    classifier = FakeClassifier()
    classifier.input_result = InputReplyClassification(InputReplyAction.UNRELATED)
    classifier.result_result = RunResultReplyClassification(
        RunResultReplyAction.UNRELATED
    )

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        "same reply",
        snapshot.context,
        message_id=MESSAGE_ID,
        created_at=NOW,
    )

    assert result.message == replay_message
    assert result.handled is True
    assert result.answer_expected is False
    assert result.inserted is False
    assert result.replayed is True
    assert classifier.calls == []
    assert store.mutation_calls == []


@pytest.mark.asyncio
async def test_durable_ordinary_replay_skips_reclassification() -> None:
    snapshot = result_snapshot()
    replay_message = user_message("thanks")
    store = FakeStore(
        AgentReplySubjectSnapshot(
            context=snapshot.context,
            source_message_id=snapshot.source_message_id,
            source_content=snapshot.source_content,
            interaction_kind=None,
            allowed_decisions=(),
            recovered_without_details=False,
            replay=AgentReplyReplay(message=replay_message, handled=False),
        )
    )
    classifier = FakeClassifier()
    classifier.result_result = RunResultReplyClassification(
        RunResultReplyAction.CONTINUE
    )

    result = await AgentReplyService(store, classifier).handle(
        FISH_ID,
        SESSION_ID,
        "thanks",
        snapshot.context,
        message_id=MESSAGE_ID,
    )

    assert result.handled is False
    assert result.answer_expected is True
    assert result.inserted is False
    assert result.replayed is True
    assert classifier.calls == []
    assert store.mutation_calls == []


@pytest.mark.asyncio
async def test_llm_reply_classifier_uses_bounded_identifier_free_exact_wire() -> None:
    client = FakeJsonClient(
        '{"action":"answer"}',
        '{"action":"changes_requested"}',
    )
    classifier = LlmReplyClassifier(client)

    input_result = await classifier.classify_input(
        request="Which branch?",
        content="Use release/next",
    )
    result_result = await classifier.classify_run_result(
        result="The test failed.",
        content="Please fix the assertion.",
    )

    assert input_result == InputReplyClassification(InputReplyAction.ANSWER)
    assert result_result == RunResultReplyClassification(
        RunResultReplyAction.CHANGES_REQUESTED
    )
    assert len(client.calls) == 2
    input_payload = json.loads(client.calls[0][1]["content"])
    result_payload = json.loads(client.calls[1][1]["content"])
    assert input_payload == {
        "request": "Which branch?",
        "reply": "Use release/next",
    }
    assert result_payload == {
        "result": "The test failed.",
        "reply": "Please fix the assertion.",
    }
    wire = json.dumps(client.calls, ensure_ascii=False, default=str)
    assert str(FISH_ID) not in wire
    assert str(SESSION_ID) not in wire
    assert str(SUBJECT_ID) not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        "{}",
        "[]",
        '{"action":"answer","extra":true}',
        '{"action":"answer","action":"unrelated"}',
        '{"action":"unknown"}',
        '{"action":""}',
        "not-json",
    ],
)
async def test_llm_reply_classifier_rejects_non_exact_output(wire: str) -> None:
    classifier = LlmReplyClassifier(FakeJsonClient(wire))

    with pytest.raises(AgentReplyClassificationError):
        await classifier.classify_input(request="Question?", content="Answer")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "reply"),
    [
        ("", "answer"),
        ("question", "   "),
        ("x" * 32_001, "answer"),
        ("question", "x" * 32_001),
    ],
)
async def test_llm_reply_classifier_rejects_blank_or_oversized_prompt(
    source: str,
    reply: str,
) -> None:
    client = FakeJsonClient('{"action":"answer"}')
    classifier = LlmReplyClassifier(client)

    with pytest.raises(AgentReplyClassificationError):
        await classifier.classify_input(request=source, content=reply)

    assert client.calls == []


@pytest.mark.asyncio
async def test_llm_reply_classifier_provider_failure_is_fail_closed() -> None:
    client = FakeJsonClient()
    client.error = RuntimeError("provider unavailable")

    with pytest.raises(AgentReplyClassificationError, match="failed"):
        await LlmReplyClassifier(client).classify_run_result(
            result="done",
            content="continue",
        )


@pytest.mark.asyncio
async def test_llm_reply_classifier_rejects_oversized_completion() -> None:
    client = FakeJsonClient('{"action":"answer"}' + " " * 1024)

    with pytest.raises(AgentReplyClassificationError):
        await LlmReplyClassifier(client).classify_input(
            request="Question?",
            content="Answer",
        )
