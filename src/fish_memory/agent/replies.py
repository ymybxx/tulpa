"""Exact, locally transactional replies to Agent interactions and results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Protocol, TypeVar
from uuid import UUID

from fish_memory.agent.models import (
    AgentInteraction,
    AgentInteractionKind,
    AgentMessageSubjectKind,
    AgentRun,
)
from fish_memory.agent.routing import AgentInteractionReplyUnrelatedError
from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.models import Message

_MAX_CLASSIFIER_TEXT_CHARS = 32_000
_MAX_CLASSIFIER_COMPLETION_BYTES = 1_024


class AgentReplyConflictError(RuntimeError):
    """Raised when an exact reply subject is stale, invalid, or already consumed."""


class AgentReplyClassificationError(AgentReplyConflictError):
    """Raised when restricted reply classification cannot be trusted."""


@dataclass(frozen=True, slots=True)
class AgentReplyContext:
    kind: AgentMessageSubjectKind
    id: UUID


def reply_context_fingerprint(context: AgentReplyContext | None) -> str:
    """Return a transcript-free exact fingerprint for voice commit tombstones."""

    if context is None:
        payload = b"reply-context:none"
    else:
        payload = f"reply-context:{context.kind.value}:{context.id}".encode()
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentReplyMutation:
    message: Message
    inserted: bool
    replayed: bool
    handled: bool
    interaction: AgentInteraction | None = None
    run: AgentRun | None = None


@dataclass(frozen=True, slots=True)
class AgentReplyReplay:
    message: Message
    handled: bool


@dataclass(frozen=True, slots=True)
class AgentReplySubjectSnapshot:
    """Bounded classifier input frozen before any model await."""

    context: AgentReplyContext
    source_message_id: UUID
    source_content: str
    interaction_kind: AgentInteractionKind | None
    allowed_decisions: tuple[str, ...]
    recovered_without_details: bool
    replay: AgentReplyReplay | None


@dataclass(frozen=True, slots=True)
class PreparedAgentInteractionReply:
    """Read-only reply classification; only commit may consume its subject."""

    fish_id: UUID
    session_id: UUID
    content: str
    snapshot: AgentReplySubjectSnapshot
    approval_choice: str | None = None
    approval_resolve_all: bool = False


@dataclass(frozen=True, slots=True)
class AgentReplyResult:
    message: Message
    handled: bool
    inserted: bool
    replayed: bool

    @property
    def answer_expected(self) -> bool:
        return not self.handled


class AgentReplyDisposition(StrEnum):
    ORDINARY = "ordinary"
    APPROVAL = "approval"
    INPUT = "input"
    RESULT = "result"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class PreparedAgentReply:
    """One exact reply plan, prepared without consuming or dispatching it."""

    fish_id: UUID
    session_id: UUID
    content: str
    snapshot: AgentReplySubjectSnapshot
    disposition: AgentReplyDisposition
    approval_choice: str | None = None
    approval_resolve_all: bool = False
    result_action: RunResultReplyAction | None = None
    message_id: UUID | None = None

    @property
    def handled(self) -> bool:
        if self.disposition is AgentReplyDisposition.REPLAY:
            replay = self.snapshot.replay
            if replay is None:
                raise AgentReplyConflictError("prepared replay has no disposition")
            return replay.handled
        return self.disposition is not AgentReplyDisposition.ORDINARY

    @property
    def requires_answer(self) -> bool:
        return not self.handled


class InputReplyAction(StrEnum):
    ANSWER = "answer"
    UNRELATED = "unrelated"


@dataclass(frozen=True, slots=True)
class InputReplyClassification:
    action: InputReplyAction


class RunResultReplyAction(StrEnum):
    ACKNOWLEDGE = "acknowledge"
    CONTINUE = "continue"
    RETRY = "retry"
    CHANGES_REQUESTED = "changes_requested"
    UNRELATED = "unrelated"


@dataclass(frozen=True, slots=True)
class RunResultReplyClassification:
    action: RunResultReplyAction


class ReplyClassifier(Protocol):
    async def classify_input(
        self,
        *,
        request: str,
        content: str,
    ) -> InputReplyClassification: ...

    async def classify_run_result(
        self,
        *,
        result: str,
        content: str,
    ) -> RunResultReplyClassification: ...


_INPUT_CLASSIFIER_PROMPT = """Classify whether the reply answers the shown
input request.
Return exactly one JSON object with only action.
action must be answer or unrelated.
Do not infer any task, user, session, or subject identity."""

_RESULT_CLASSIFIER_PROMPT = """Classify the reply to the shown completed run result.
Return exactly one JSON object with only action.
action must be acknowledge, continue, retry, changes_requested, or unrelated.
Do not infer any task, user, session, or subject identity."""

_ActionT = TypeVar("_ActionT", InputReplyAction, RunResultReplyAction)


class LlmReplyClassifier:
    """One strict JSON classifier over bounded, user-visible text only."""

    def __init__(self, client: JsonCompletionClient) -> None:
        self._client = client

    async def classify_input(
        self,
        *,
        request: str,
        content: str,
    ) -> InputReplyClassification:
        payload = {
            "request": _classifier_text("request", request),
            "reply": _classifier_text("reply", content),
        }
        action = await self._classify(
            _INPUT_CLASSIFIER_PROMPT,
            payload,
            InputReplyAction,
        )
        return InputReplyClassification(action)

    async def classify_run_result(
        self,
        *,
        result: str,
        content: str,
    ) -> RunResultReplyClassification:
        payload = {
            "result": _classifier_text("result", result),
            "reply": _classifier_text("reply", content),
        }
        action = await self._classify(
            _RESULT_CLASSIFIER_PROMPT,
            payload,
            RunResultReplyAction,
        )
        return RunResultReplyClassification(action)

    async def _classify(
        self,
        system_prompt: str,
        payload: Mapping[str, str],
        action_type: type[_ActionT],
    ) -> _ActionT:
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            completion = await self._client.complete_json(messages)
            if (
                not isinstance(completion.content, str)
                or len(completion.content.encode("utf-8"))
                > _MAX_CLASSIFIER_COMPLETION_BYTES
            ):
                raise ValueError("reply classification completion is too large")
            value = json.loads(
                completion.content,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            if not isinstance(value, dict) or set(value) != {"action"}:
                raise ValueError("reply classification must contain only action")
            action = value["action"]
            if not isinstance(action, str) or not action:
                raise ValueError("reply classification action must be non-empty")
            return action_type(action)
        except Exception as error:
            raise AgentReplyClassificationError(
                "reply classification failed closed"
            ) from error


class AgentReplyStore(Protocol):
    async def load_reply_subject(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        *,
        message_id: UUID | None = None,
        content: str | None = None,
    ) -> AgentReplySubjectSnapshot: ...

    async def insert_and_bind_approval_response(
        self,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        decision: Mapping[str, object] | str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
        adopt_existing: bool = False,
    ) -> AgentReplyMutation: ...

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
    ) -> AgentReplyMutation: ...

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
    ) -> AgentReplyMutation: ...

    async def insert_validated_ordinary_reply(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyMutation: ...


_DENY_PHRASES = frozenset(
    {
        "拒绝",
        "不同意",
        "不批准",
        "不允许",
        "否",
        "deny",
        "reject",
        "no",
        "do not approve",
        "don't approve",
    }
)
_ALLOW_ONCE_PHRASES = frozenset(
    {
        "允许",
        "同意",
        "批准",
        "允许这次",
        "批准这次",
        "仅允许这一次",
        "只允许这一次",
        "只批准这一次",
        "allow",
        "approve",
        "allow once",
        "approve once",
        "yes",
        "yes, approve",
    }
)
_ALLOW_SESSION_PHRASES = frozenset(
    {
        "本会话允许",
        "仅本会话允许",
        "本次会话允许",
        "allow for this session",
        "approve for this session",
    }
)
_ALLOW_ALWAYS_PHRASES = frozenset(
    {
        "始终允许",
        "总是允许",
        "always allow",
        "always approve",
    }
)
_ALLOW_ALL_PHRASES = frozenset(
    {
        "允许所有待处理请求",
        "批准所有待处理请求",
        "allow all pending requests",
        "approve all pending requests",
    }
)


class AgentReplyService:
    """Classify against one frozen subject, then consume it with a Store CAS."""

    def __init__(
        self,
        store: AgentReplyStore,
        classifier: ReplyClassifier,
    ) -> None:
        self._store = store
        self._classifier = classifier

    async def handle(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        context: AgentReplyContext,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyResult:
        prepared = await self.prepare_reply(
            fish_id,
            session_id,
            content,
            context,
            message_id=message_id,
        )
        return await self.commit_prepared_reply(
            prepared, message_id=message_id, created_at=created_at
        )

    async def prepare_reply(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        context: AgentReplyContext,
        *,
        message_id: UUID | None = None,
    ) -> PreparedAgentReply:
        """Classify against a frozen subject without inserting any Message."""

        frozen = AgentReplyContext(context.kind, context.id)
        snapshot = await self._store.load_reply_subject(
            frozen,
            fish_id,
            session_id,
            message_id=message_id,
            content=content,
        )
        if snapshot.context != frozen:
            raise AgentReplyConflictError("reply subject identity changed")
        if snapshot.replay is not None:
            return PreparedAgentReply(
                fish_id,
                session_id,
                content,
                snapshot,
                AgentReplyDisposition.REPLAY,
                message_id=message_id,
            )
        disposition = AgentReplyDisposition.ORDINARY
        choice: str | None = None
        resolve_all = False
        result_action: RunResultReplyAction | None = None
        if frozen.kind is AgentMessageSubjectKind.INTERACTION:
            if snapshot.interaction_kind is AgentInteractionKind.APPROVAL:
                decision = _parse_approval(content)
                if decision is not None:
                    choice = str(decision["choice"])
                    if snapshot.recovered_without_details and choice != "deny":
                        raise AgentReplyConflictError(
                            "recovered approval without details can only be denied"
                        )
                    if choice not in snapshot.allowed_decisions:
                        raise AgentReplyConflictError(
                            "approval decision is not allowed"
                        )
                    resolve_all = bool(decision["resolve_all"])
                    disposition = AgentReplyDisposition.APPROVAL
            elif snapshot.interaction_kind is AgentInteractionKind.INPUT:
                classification = await self._classifier.classify_input(
                    request=snapshot.source_content, content=content
                )
                if classification.action is InputReplyAction.ANSWER:
                    disposition = AgentReplyDisposition.INPUT
                elif classification.action is not InputReplyAction.UNRELATED:
                    raise AgentReplyConflictError("invalid input reply classification")
            else:
                raise AgentReplyConflictError("reply subject is not an interaction")
        elif frozen.kind is AgentMessageSubjectKind.RUN_RESULT:
            classification = await self._classifier.classify_run_result(
                result=snapshot.source_content, content=content
            )
            result_action = classification.action
            if result_action in {
                RunResultReplyAction.CONTINUE,
                RunResultReplyAction.RETRY,
                RunResultReplyAction.CHANGES_REQUESTED,
            }:
                disposition = AgentReplyDisposition.RESULT
            elif result_action not in {
                RunResultReplyAction.ACKNOWLEDGE,
                RunResultReplyAction.UNRELATED,
            }:
                raise AgentReplyConflictError("invalid result reply classification")
        else:  # pragma: no cover - enum construction rejects this at API boundary
            raise AgentReplyConflictError("unsupported reply subject kind")
        return PreparedAgentReply(
            fish_id,
            session_id,
            content,
            snapshot,
            disposition,
            approval_choice=choice,
            approval_resolve_all=resolve_all,
            result_action=result_action,
            message_id=message_id,
        )

    async def commit_prepared_reply(
        self,
        prepared: PreparedAgentReply,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyResult:
        """Revalidate one subject and commit its plan without another classifier."""

        if (
            message_id is not None
            and prepared.message_id is not None
            and message_id != prepared.message_id
        ):
            raise AgentReplyConflictError("prepared reply Message identity changed")
        message_id = message_id or prepared.message_id
        snapshot = prepared.snapshot
        if snapshot.replay is not None:
            return _reply_replay_result(snapshot.replay)
        current = await self._store.load_reply_subject(
            snapshot.context,
            prepared.fish_id,
            prepared.session_id,
            message_id=message_id,
            content=prepared.content,
        )
        if current.context != snapshot.context:
            raise AgentReplyConflictError("prepared reply subject identity changed")
        if current.replay is not None:
            return _reply_replay_result(current.replay)
        if current != snapshot:
            raise AgentReplyConflictError("prepared reply subject changed")
        kind = prepared.disposition
        if kind is AgentReplyDisposition.APPROVAL:
            if prepared.approval_choice is None:
                raise AgentReplyConflictError("prepared approval has no decision")
            mutation = await self._store.insert_and_bind_approval_response(
                snapshot.context.id,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                {
                    "choice": prepared.approval_choice,
                    "resolve_all": prepared.approval_resolve_all,
                },
                message_id=message_id,
                created_at=created_at,
            )
        elif kind is AgentReplyDisposition.INPUT:
            mutation = await self._store.insert_and_resolve_input(
                snapshot.context.id,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                message_id=message_id,
                created_at=created_at,
            )
        elif kind is AgentReplyDisposition.RESULT:
            if prepared.result_action is None:
                raise AgentReplyConflictError("prepared result has no action")
            mutation = await self._store.insert_and_create_result_followup(
                snapshot.context.id,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                prepared.result_action.value,
                message_id=message_id,
                created_at=created_at,
            )
        elif kind is AgentReplyDisposition.ORDINARY:
            mutation = await self._ordinary(
                snapshot.context,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                message_id=message_id,
                created_at=created_at,
            )
        else:
            raise AgentReplyConflictError("prepared reply has no commit disposition")
        return AgentReplyResult(
            message=mutation.message,
            handled=mutation.handled,
            inserted=mutation.inserted,
            replayed=mutation.replayed,
        )

    async def continue_interaction(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        interaction_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID,
        content: str,
    ) -> AgentRun | None:
        """Adopt one already-persisted T-lane message as an Interaction reply."""

        prepared = await self.prepare_interaction_reply(
            fish_id=fish_id,
            session_id=session_id,
            interaction_id=interaction_id,
            content=content,
        )
        return await self.commit_prepared_interaction(
            prepared,
            source_message_id=source_message_id,
            confirmation_message_id=confirmation_message_id,
        )

    async def prepare_interaction_reply(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        interaction_id: UUID,
        content: str,
    ) -> PreparedAgentInteractionReply:
        """Validate and classify one exact interaction without a Store mutation."""

        if not content.strip():
            raise AgentReplyConflictError("interaction reply must not be blank")
        context = AgentReplyContext(
            AgentMessageSubjectKind.INTERACTION,
            interaction_id,
        )
        snapshot = await self._store.load_reply_subject(
            context,
            fish_id,
            session_id,
        )
        if snapshot.context != context or snapshot.replay is not None:
            raise AgentReplyConflictError("interaction route subject changed")
        if snapshot.interaction_kind is AgentInteractionKind.APPROVAL:
            decision = _parse_approval(content)
            if decision is None:
                raise AgentReplyConflictError(
                    "selected approval route does not contain a decision"
                )
            choice = str(decision["choice"])
            if snapshot.recovered_without_details and choice != "deny":
                raise AgentReplyConflictError(
                    "recovered approval without details can only be denied"
                )
            if choice not in snapshot.allowed_decisions:
                raise AgentReplyConflictError("approval decision is not allowed")
            return PreparedAgentInteractionReply(
                fish_id,
                session_id,
                content,
                snapshot,
                approval_choice=choice,
                approval_resolve_all=bool(decision["resolve_all"]),
            )
        if snapshot.interaction_kind is AgentInteractionKind.INPUT:
            classification = await self._classifier.classify_input(
                request=snapshot.source_content,
                content=content,
            )
            if classification.action is InputReplyAction.UNRELATED:
                raise AgentInteractionReplyUnrelatedError(
                    "selected input Interaction is unrelated"
                )
            if classification.action is not InputReplyAction.ANSWER:
                raise AgentReplyConflictError("invalid input reply classification")
            return PreparedAgentInteractionReply(fish_id, session_id, content, snapshot)
        raise AgentReplyConflictError("reply subject is not an interaction")

    async def commit_prepared_interaction(
        self,
        prepared: PreparedAgentInteractionReply,
        *,
        source_message_id: UUID,
        confirmation_message_id: UUID,
    ) -> AgentRun | None:
        """Consume the frozen disposition through the existing transactional CAS."""

        snapshot = prepared.snapshot
        current = await self._store.load_reply_subject(
            snapshot.context,
            prepared.fish_id,
            prepared.session_id,
        )
        if current != snapshot or current.replay is not None:
            raise AgentReplyConflictError("prepared interaction reply subject changed")
        if snapshot.interaction_kind is AgentInteractionKind.APPROVAL:
            if prepared.approval_choice is None:
                raise AgentReplyConflictError("prepared approval has no decision")
            mutation = await self._store.insert_and_bind_approval_response(
                snapshot.context.id,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                {
                    "choice": prepared.approval_choice,
                    "resolve_all": prepared.approval_resolve_all,
                },
                message_id=source_message_id,
                adopt_existing=True,
            )
        elif snapshot.interaction_kind is AgentInteractionKind.INPUT:
            mutation = await self._store.insert_and_resolve_input(
                snapshot.context.id,
                prepared.fish_id,
                prepared.session_id,
                prepared.content,
                message_id=source_message_id,
                confirmation_message_id=confirmation_message_id,
                adopt_existing=True,
            )
        else:
            raise AgentReplyConflictError("reply subject is not an interaction")
        if not mutation.handled:
            raise AgentReplyConflictError("interaction route did not bind its reply")
        return mutation.run

    async def _ordinary(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None,
        created_at: datetime | None,
    ) -> AgentReplyMutation:
        return await self._store.insert_validated_ordinary_reply(
            context,
            fish_id,
            session_id,
            content,
            message_id=message_id,
            created_at=created_at,
        )


def _reply_replay_result(replay: AgentReplyReplay) -> AgentReplyResult:
    return AgentReplyResult(
        message=replay.message,
        handled=replay.handled,
        inserted=False,
        replayed=True,
    )


def _parse_approval(content: str) -> dict[str, object] | None:
    normalized = content.strip().casefold().rstrip("。.!！?？").strip()
    if normalized in _DENY_PHRASES:
        return {"choice": "deny", "resolve_all": False}
    if normalized in _ALLOW_SESSION_PHRASES:
        return {"choice": "session", "resolve_all": False}
    if normalized in _ALLOW_ALWAYS_PHRASES:
        return {"choice": "always", "resolve_all": False}
    if normalized in _ALLOW_ALL_PHRASES:
        return {"choice": "once", "resolve_all": True}
    if normalized in _ALLOW_ONCE_PHRASES:
        return {"choice": "once", "resolve_all": False}
    return None


def _classifier_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentReplyClassificationError(f"{name} must not be blank")
    if len(value) > _MAX_CLASSIFIER_TEXT_CHARS:
        raise AgentReplyClassificationError(
            f"{name} exceeds {_MAX_CLASSIFIER_TEXT_CHARS} characters"
        )
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value
