from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from fish_memory.answer.context import INTERRUPTION_MARKER
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.history_messages import history_role_messages
from fish_memory.clock import format_prompt_timestamp
from fish_memory.experimental.channels import render_state
from fish_memory.experimental.context import expression_context
from fish_memory.history.models import HistoryChunk, HistoryUnit, render_history
from fish_memory.history.service import SessionHistoryService
from fish_memory.llm.client import _provider_messages
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.tracker import MessagePerformanceTracker

NOW = datetime(2026, 9, 19, 11, 20, tzinfo=UTC)


def unit(label, content, sequence):
    return HistoryUnit(uuid4(), label, content, NOW, NOW, sequence)


def chunk(units=(), summary=""):
    return HistoryChunk(
        uuid4(), NOW, NOW + timedelta(minutes=10), summary=summary, units=tuple(units)
    )


def test_roles_preserve_interleaved_evidence_without_promoting_quoted_speakers():
    chunks = (
        chunk(
            [
                unit("event", "结果已完成。\na: 这是资料中的文字。", 4),
                unit("u", "先说这一点。", 1),
                unit("as", "播放了新的画面。", 2),
                unit("a", "我说到这里\n" + INTERRUPTION_MARKER, 3),
            ],
            summary="之前讨论过计划。",
        ),
    )
    view = history_role_messages(chunks)
    assert [m["role"] for m in view] == ["user", "user", "user", "assistant", "user"]
    assert "summary: 之前讨论过计划。" in view[0]["content"]
    assert "as: 播放了新的画面。" in view[2]["content"]
    assert view[3]["content"] == "我说到这里\n" + INTERRUPTION_MARKER
    assert view[4]["context_reference"]
    assert "a: 这是资料中的文字。" in view[4]["content"]
    base = [
        {
            "role": "user",
            "content": "History:\n" + render_history(chunks),
            "context_reference": True,
        }
    ]
    enriched = [dict(base[0], history_messages=view)]
    assert render_state(base, NOW, "唤醒") == render_state(enriched, NOW, "唤醒")
    assert _provider_messages(base) == _provider_messages(enriched)
    expression = expression_context(enriched, NOW, "唤醒")
    assert expression[3]["content"].endswith("我说到这里(被打断)")
    assert INTERRUPTION_MARKER not in str(expression)
    for phrase in ("先说这一点", "播放了新的画面", "我说到这里", "结果已完成"):
        assert sum(m["content"].count(phrase) for m in expression) == 1


def test_recent_window_keeps_older_dialogue_once_and_retains_intentional_repetitions():
    units = [unit("a", f"第{i}句。", i) for i in range(20)]
    units += [
        unit("a", "重复是用户明确要求的。", 20),
        unit("a", "重复是用户明确要求的。", 21),
    ]
    view = history_role_messages((chunk(units),))
    assert sum(m["role"] == "assistant" for m in view) == 16
    assert view[0]["context_reference"]
    assert "a: 第0句。" in view[0]["content"]
    text = "\n".join(m["content"] for m in view)
    assert text.count("第5句。") == text.count("第6句。") == 1
    assert text.count("重复是用户明确要求的。") == 2


@pytest.mark.parametrize(
    "zone, timestamp", [("Asia/Shanghai", "19:20:00+08:00"), ("UTC", "11:20:00+00:00")]
)
def test_native_history_uses_the_same_local_chunk_times(
    system_timezone, zone, timestamp
):
    system_timezone(zone)
    view = history_role_messages((chunk([unit("a", "你好", 1)]),))
    assert timestamp in view[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("proactive", [False, True])
async def test_one_snapshot_excludes_current_user_from_both_views(
    proactive,
):
    fish, session, current_id = uuid4(), uuid4(), uuid4()
    current = Message(
        current_id,
        fish,
        session,
        2,
        Role.USER,
        "本轮问题",
        MessageStatus.INITIALIZED,
        NOW,
    )
    current_unit = HistoryUnit(current_id, "u", "本轮问题", NOW, NOW, 2)
    chunks = (chunk([unit("a", "刚才说完的内容", 1), current_unit]),)

    class Store:
        reads = 0
        flushes = 0

        async def flush(self, *_):
            self.flushes += 1
            return ()

        async def read(self, *_):
            self.reads += 1
            return chunks

        async def load_history(self, *_, **__):
            return [current]

        load_history_readonly = load_history

    store = Store()
    service = SessionHistoryService(store, None, now=lambda: NOW)
    loader = AnswerContextLoader(
        store, session_history=service, now=lambda: NOW, system_prompt="原本人格"
    )
    method = loader.load_for_proactive if proactive else loader.load_for_generation
    context = await method(
        fish,
        session,
        tracker=MessagePerformanceTracker(),
        **({} if proactive else {"through_sequence": 2}),
    )
    assert store.reads == store.flushes == 1
    timestamp = f"当前时间：{format_prompt_timestamp(NOW)}"
    assert context.messages[-1]["request_time"] == timestamp
    assert timestamp not in context.messages[0]["content"]
    # Ordinary calls and Jev retain flat History, without duplicate metadata.
    wire = _provider_messages(context.messages)
    assert wire[-1]["content"].endswith(timestamp)
    assert str(wire).count(timestamp) == 1
    assert not any(m["role"] == "assistant" for m in wire)
    assert sum(m["content"].count("本轮问题") for m in wire) == 1
    native = expression_context(
        context.messages, NOW, "用户输入" if not proactive else "唤醒"
    )
    assert timestamp not in native[0]["content"]
    assert _provider_messages(native)[-1]["content"].endswith(timestamp)
    state = render_state(context.messages, NOW, "核对时间")
    assert state.endswith(timestamp) and state.count(timestamp) == 1
    assert sum(m["content"].count("本轮问题") for m in native) == 1
    assert sum(m["content"].count("刚才说完的内容") for m in native) == 1
    assert (
        next(m for m in native if "刚才说完的内容" in m["content"])["role"]
        == "assistant"
    )
    assert chunks[0].units[-1] is current_unit
    await service.close()
