from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import asyncpg
import pytest
from pydantic import SecretStr

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.voice_draft_types import VoiceDraftKey, voice_answer_message_id
from fish_memory.answer.voice_drafts import VoiceDraftLease
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.prepared import ExperimentalPreparedAnswer
from fish_memory.experimental.service import ExperimentalService
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker

NOW = datetime(2026, 9, 19, 9, 45, tzinfo=UTC)


class SelectSpeak:
    async def decide(self, configuration, state, available):
        return "s"

    async def close(self):
        pass


class SelectedChat:
    invalid = True

    async def stream_json(self, messages, **kwargs):
        if self.invalid:
            yield ChatStreamChunk(content='{"m":"neutral","a":"","e":0}')
        else:
            yield ChatStreamChunk(
                content='{"m":"happy","a":"现在能正常回答了。","e":0}'
            )


@pytest.mark.asyncio
@pytest.mark.pg0
async def test_invalid_experimental_reply_releases_real_session_constraint(
    postgres_pool,
):
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(cast(asyncpg.Pool, postgres_pool))
    fish = uuid4()
    first = await store.insert_user_message(fish, "这是一轮测试。")
    chat = SelectedChat()
    experiment = ExperimentalService(
        ExperimentalConfiguration(
            enabled=True, chat_pick_enabled=True, api_key=SecretStr("test-only")
        ),
        judge=SelectSpeak(),
        chat=chat,
        reasoning=chat,
        recall=None,
        observations=SimpleNamespace(),
        now=lambda: NOW,
        search_enabled=False,
        recall_token_budget=5000,
    )
    coordinator = AnswerCoordinator(
        store, chat, experiment=experiment, system_prompt="你陪伴用户。"
    )
    tracker = MessagePerformanceTracker()
    prepared = ExperimentalPreparedAnswer(
        experiment,
        fish,
        LoadedAnswerContext(
            [
                {"role": "system", "content": "你陪伴用户。"},
                {"role": "user", "content": first.content},
            ],
            (),
        ),
        PromotableAnswerSink(),
        tracker,
    )
    lease = VoiceDraftLease(
        VoiceDraftKey(fish, first.session_id, first.id),
        1,
        "candidate",
        "input",
        NOW,
        prepared,
    )
    answer_id = voice_answer_message_id(first.id)
    try:
        await coordinator.handle_user_message(
            fish,
            first.session_id,
            first.sequence,
            message_id=first.id,
            answer_message_id=answer_id,
            source_message=first,
            tracker=tracker,
            draft_lease=lease,
        )
        runtime = coordinator.get_runtime(fish, first.session_id)
        await runtime.llm_task
        row = await postgres_pool.fetchrow(
            "SELECT status,content,finished_at FROM messages WHERE id=$1", answer_id
        )
        assert row["status"] == "failed"
        assert row["content"] == "" and row["finished_at"] is not None

        chat.invalid = False
        second = await store.insert_user_message(
            fish, "开启对话。", session_id=first.session_id
        )
        await coordinator.handle_user_message(
            fish,
            second.session_id,
            second.sequence,
            message_id=second.id,
            source_message=second,
            tracker=MessagePerformanceTracker(),
        )
        await runtime.llm_task
        assert runtime.status is MessageStatus.COMPLETED
        assert runtime.content == "现在能正常回答了。"
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM messages "
                "WHERE session_id=$1 AND status='generating'",
                first.session_id,
            )
            == 0
        )
    finally:
        await coordinator.close()
        await experiment.close()
