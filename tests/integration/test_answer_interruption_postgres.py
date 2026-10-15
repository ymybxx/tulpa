import asyncio
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker
from tests.answer.fakes import ControlledChatClient


@pytest.mark.asyncio
async def test_interrupt_cancels_terminal_transaction_and_next_answer_can_start(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))

    class ObservedStore(MessagesStore):
        completion_started = asyncio.Event()

        async def finish_answer(
            self,
            message_id: UUID,
            fish_id: UUID,
            *,
            content: str,
            requested_status: MessageStatus,
            failure: str | None = None,
        ) -> MessageStatus:
            if requested_status is MessageStatus.COMPLETED:
                self.completion_started.set()
            return await super().finish_answer(
                message_id,
                fish_id,
                content=content,
                requested_status=requested_status,
                failure=failure,
            )

    store = ObservedStore(postgres_pool)
    fish_id = uuid4()
    user = await store.insert_user_message(fish_id, "测试")
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")
    interrupt_task: asyncio.Task[None] | None = None
    try:
        await coordinator.handle_user_message(
            fish_id,
            user.session_id,
            sequence=user.sequence,
            message_id=user.id,
            tracker=MessagePerformanceTracker(),
            playback_owner="companion",
        )
        runtime = coordinator.get_runtime(fish_id, user.session_id)
        answer_id = runtime.answer_message_id
        generation_task = runtime.llm_task
        assert answer_id is not None and generation_task is not None
        await llm.wait_for_streams(1)
        await llm.emit(0, "我得好好捋捋哦，后半句")
        async with asyncio.timeout(2), runtime.condition:
            await runtime.condition.wait_for(
                lambda: runtime.content == "我得好好捋捋哦，后半句"
            )
        await coordinator.update_progress(fish_id, answer_id, 7)

        async with postgres_pool.acquire() as blocking_connection:
            async with blocking_connection.transaction():
                await blocking_connection.fetchrow(
                    "SELECT id FROM messages WHERE id = $1 FOR UPDATE", answer_id
                )
                await llm.finish(0)
                await asyncio.wait_for(store.completion_started.wait(), timeout=2)
                # Cancel a real blocked finish_answer transaction, not just the
                # model stream. The next transaction must release the unique slot.
                async with asyncio.timeout(2):
                    while not await postgres_pool.fetchval(  # noqa: ASYNC110 - observe a PostgreSQL lock
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_stat_activity
                            WHERE datname = current_database()
                              AND wait_event_type = 'Lock'
                              AND query LIKE '%context_through_sequence%'
                              AND query LIKE '%FOR UPDATE%'
                        )
                        """
                    ):
                        await asyncio.sleep(0.01)
                interrupt_task = asyncio.create_task(
                    coordinator.interrupt_answer(fish_id, answer_id, 7)
                )
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(generation_task), timeout=2)
        await asyncio.wait_for(interrupt_task, timeout=2)

        row = await postgres_pool.fetchrow(
            "SELECT status, content, consumed_offset FROM messages WHERE id = $1",
            answer_id,
        )
        assert dict(row) == {
            "status": "interrupted",
            "content": "我得好好捋捋哦",
            "consumed_offset": 7,
        }
        next_user = await store.insert_user_message(
            fish_id, "继续", session_id=user.session_id
        )
        await coordinator.handle_user_message(
            fish_id,
            next_user.session_id,
            sequence=next_user.sequence,
            message_id=next_user.id,
            tracker=MessagePerformanceTracker(),
        )
        await llm.wait_for_streams(2)
        await llm.emit(1, "可以继续")
        await llm.finish(1)
        assert runtime.llm_task is not None
        await asyncio.wait_for(runtime.llm_task, timeout=2)
        assert runtime.status is MessageStatus.COMPLETED
        assert (
            await postgres_pool.fetchval(
                """
            SELECT count(*) FROM messages
            WHERE session_id = $1 AND status = 'generating'
            """,
                user.session_id,
            )
            == 0
        )
    finally:
        if interrupt_task is not None:
            await asyncio.gather(interrupt_task, return_exceptions=True)
        await coordinator.close()
