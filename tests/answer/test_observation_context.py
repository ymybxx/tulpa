from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.context import latest_user_content
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.lane_context import build_recall_lane_messages
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.observations.context import ObservationContextLoader
from fish_memory.performance.tracker import MessagePerformanceTracker

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")
NOW = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def user_message(content: str, sequence: int) -> Message:
    return Message(
        id=UUID(int=sequence),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=sequence,
        role=Role.USER,
        content=content,
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )


class HistoryStore:
    def __init__(self, rows: list[Message]) -> None:
        self.rows = rows

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        del fish_id, session_id, limit
        return [row for row in self.rows if row.sequence < before_sequence]

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
        appended_history_token_count: int | None = None,
    ) -> list[Message]:
        del fish_id, session_id, before_sequence, limit, appended_history_token_count
        return list(self.rows)


class FixedObservationContextLoader:
    def __init__(self, context: str) -> None:
        self.context = context
        self.calls: list[tuple[UUID, str]] = []

    async def load(self, fish_id: UUID, question: str) -> str:
        self.calls.append((fish_id, question))
        return self.context


class FailingObservationContextLoader:
    async def load(self, fish_id: UUID, question: str) -> str:
        del fish_id, question
        raise RuntimeError("observation database unavailable")


@pytest.mark.asyncio
async def test_draft_and_committed_generation_use_same_observation_context_path() -> (
    None
):
    question = "我现在在电脑上做什么？"
    observation_context = "\n\n# 你留意到的事情\nstructured observation"
    history = HistoryStore([user_message("之前的问题", 1)])
    observations = FixedObservationContextLoader(observation_context)
    loader = AnswerContextLoader(
        history,
        observation_context_loader=cast(ObservationContextLoader, observations),
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    draft = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content=question,
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )
    history.rows.append(user_message(question, 2))
    committed = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=2,
        tracker=MessagePerformanceTracker(),
    )

    assert draft.messages == committed.messages
    assert "structured observation" not in draft.messages[0]["content"]
    assert "你看到和听到的世界" in draft.messages[0]["content"]
    assert draft.messages[-2] == {
        "role": "user",
        "content": "# 你留意到的事情\nstructured observation",
        "context_reference": True,
    }
    assert draft.messages[-1]["content"].endswith(question)
    assert all(
        "structured observation" not in message["content"]
        for message in (*draft.messages[1:-2], draft.messages[-1])
    )
    assert all(
        "structured observation" not in message["content"]
        for message in draft.recall_conversation
    )
    assert latest_user_content(draft.messages) == question
    recall_lane = build_recall_lane_messages(draft.messages)
    assert any(
        "structured observation" in message["content"] for message in recall_lane[:-1]
    )
    assert "structured observation" not in recall_lane[-1]["content"]
    assert observations.calls == [(FISH_ID, question), (FISH_ID, question)]


@pytest.mark.asyncio
async def test_observation_context_failure_is_logged_measured_and_silently_omitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracker = MessagePerformanceTracker()
    loader = AnswerContextLoader(
        HistoryStore([user_message("我刚才在电脑上做了什么？", 1)]),
        observation_context_loader=cast(
            ObservationContextLoader,
            FailingObservationContextLoader(),
        ),
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=1,
        tracker=tracker,
    )

    assert loaded.messages[0]["content"] == "base prompt"
    observation_step = next(
        step for step in tracker.snapshot() if step.name == "observation.context.load"
    )
    assert observation_step.error == "RuntimeError"
    assert observation_step.output_payload == {"included": False}
    assert "Failed to load Observation context" in caplog.text
