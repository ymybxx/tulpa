from types import SimpleNamespace
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.answer.output_sink import PromotableAnswerSink, RuntimeAnswerSink
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.speech_style import (
    CONTROL_TAG_GUIDE,
    EFFECT_TAG_GUIDE,
    with_speech_tag_rules,
)
from fish_memory.answer.sse import iter_answer_events
from fish_memory.background_messages.streams import ExpressionStream
from fish_memory.experimental.channels import render_state
from fish_memory.experimental.context import expression_context
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.tts.speech_tags import CONTROL_TAGS, EFFECT_TAGS
from tests.test_experimental_companion import NOW


@pytest.mark.asyncio
async def test_tagged_draft_promotes_clean_offsets_and_lossless_speech():
    private = PromotableAnswerSink(speech_tags=True)
    await private.append("[exci")
    assert private.snapshot().content == ""
    await private.append("ted]太好了！[laughing]")
    await private.flush()
    snapshot = private.snapshot()
    assert snapshot.content == "太好了！"
    assert snapshot.flush_offset == 4
    assert snapshot.speech_content == "[excited]太好了！[laughing]"
    runtime = FishRuntime(
        answer_message_id=UUID(int=1), status=MessageStatus.GENERATING
    )
    public = RuntimeAnswerSink(runtime, UUID(int=1), MessagePerformanceTracker())
    await private.attach(public)
    await private.append("[whispers]继续。")
    await private.flush()
    assert runtime.content == "太好了！继续。"
    assert runtime.speech_content == "[excited]太好了！[laughing][whispers]继续。"
    assert runtime.flush_offset == len(runtime.content)
    assert runtime.speech_tags


@pytest.mark.asyncio
async def test_unsupported_voice_label_never_reaches_published_text_or_tts():
    private = PromotableAnswerSink(speech_tags=True)
    await private.append("[disapp")
    assert private.snapshot().content == ""
    assert private.snapshot().speech_content == "[disapp"
    await private.append("ointed]啊？为什么我得吃肥的呀？")
    await private.flush()
    assert private.snapshot().content == "啊？为什么我得吃肥的呀？"
    assert private.snapshot().speech_content == "[disappointed]啊？为什么我得吃肥的呀？"

    runtime = FishRuntime(
        answer_message_id=UUID(int=1), status=MessageStatus.GENERATING
    )
    public = RuntimeAnswerSink(runtime, UUID(int=1), MessagePerformanceTracker())
    await private.attach(public)
    await private.append("[pout")
    await private.append("ing]还有这句。")
    await private.flush()
    assert runtime.content == "啊？为什么我得吃肥的呀？还有这句。"
    assert runtime.speech_content == (
        "[disappointed]啊？为什么我得吃肥的呀？[pouting]还有这句。"
    )
    assert private.snapshot().speech_content == runtime.speech_content


@pytest.mark.asyncio
async def test_tagged_promotion_replays_flush_boundaries_without_flushing_tail():
    private = PromotableAnswerSink(speech_tags=True)
    await private.append("正文[whis")
    await private.flush()
    await private.append("pers]后文[laugh")
    await private.flush()
    await private.append("ing]末尾[exci")
    expected = private.snapshot()
    runtime = FishRuntime(
        answer_message_id=UUID(int=1), status=MessageStatus.GENERATING
    )
    public = RuntimeAnswerSink(runtime, UUID(int=1), MessagePerformanceTracker())

    await private.attach(public)

    assert runtime.content == expected.content == "正文[whispers]后文[laughing]末尾"
    assert (
        runtime.flush_offset == expected.flush_offset == len("正文[whispers]后文[laugh")
    )
    assert runtime.speech_content == expected.speech_content
    await private.append("ted]继续。")
    await private.flush()
    assert (
        runtime.content
        == private.snapshot().content
        == ("正文[whispers]后文[laughing]末尾继续。")
    )


@pytest.mark.asyncio
async def test_tag_only_private_flush_remains_an_explicit_boundary_after_promotion():
    private = PromotableAnswerSink(speech_tags=True)
    await private.append("[laughing]")
    await private.flush()
    await private.append("[whis")
    runtime = FishRuntime(
        answer_message_id=UUID(int=1), status=MessageStatus.GENERATING
    )
    public = RuntimeAnswerSink(runtime, UUID(int=1), MessagePerformanceTracker())

    await private.attach(public)

    assert runtime.content == ""
    assert runtime.speech_content == "[laughing][whis"
    await private.append("pers]你好。")
    await private.flush()
    assert runtime.content == "你好。"


@pytest.mark.asyncio
async def test_disabled_tags_do_not_rewrite_ordinary_bracketed_text():
    sink = PromotableAnswerSink()
    await sink.append("例子：[sad]，以及[参数]。")
    await sink.flush()
    assert sink.snapshot().content == "例子：[sad]，以及[参数]。"
    assert not sink.snapshot().speech_tags


@pytest.mark.asyncio
async def test_sse_emits_tag_only_update_without_advancing_clean_offsets():
    runtime = FishRuntime(
        answer_message_id=UUID(int=1), status=MessageStatus.GENERATING, speech_tags=True
    )
    events = iter_answer_events(runtime)
    assert (await anext(events)).data["speechContent"] == ""
    sink = RuntimeAnswerSink(runtime, UUID(int=1), MessagePerformanceTracker())
    await sink.append("[excited]")
    event = await anext(events)
    assert event.event == "answer.delta"
    assert event.data["content"] == ""
    assert event.data["startOffset"] == event.data["endOffset"] == 0
    assert event.data["speechContent"] == "[excited]"
    await sink.append("你好！")
    event = await anext(events)
    assert event.data["content"] == "你好！"
    assert event.data["endOffset"] == 3
    assert event.data["speechContent"] == "[excited]你好！"
    await events.aclose()


@pytest.mark.asyncio
async def test_background_stream_keeps_history_clean_and_speech_lossless():
    stream = ExpressionStream(SimpleNamespace(text=""))
    stream.configure_speech_tags(True)
    stream.append("[sad]你好。")
    stream.append("[sighing]")
    stream.finish_text()
    stream.end()
    events = [event async for event in stream.events(0)]
    assert events[0]["text"] == stream.intent.text == "你好。"
    assert events[0]["speechContent"] == "[sad]你好。[sighing]"
    assert events[-1]["offset"] == 3


def test_voice_rules_are_generation_only_and_disabled_requests_are_unchanged():
    messages: list[ChatMessage] = [
        {"role": "system", "content": "角色设定", "speech_tags": True},
        {"role": "user", "content": "你好"},
    ]
    state = render_state(messages, NOW, "聊天")
    context = expression_context(messages, NOW, "聊天")
    with_rules = with_speech_tag_rules(context)
    assert "# 语音表达标签" not in state
    assert "# 语音表达标签" in with_rules[0]["content"]
    assert "[excited]" in with_rules[0]["content"]
    assert context[0]["content"] == "角色设定"
    plain: list[ChatMessage] = [{"role": "system", "content": "角色设定"}]
    assert with_speech_tag_rules(plain) is plain


def test_voice_rules_explain_every_supported_label_and_nothing_else():
    assert set(CONTROL_TAG_GUIDE) == set(CONTROL_TAGS)
    assert set(EFFECT_TAG_GUIDE) == set(EFFECT_TAGS)
    rules = with_speech_tag_rules(
        [{"role": "system", "content": "角色设定", "speech_tags": True}]
    )[0]["content"]
    for tag, meaning in (*CONTROL_TAG_GUIDE.items(), *EFFECT_TAG_GUIDE.items()):
        assert f"[{tag}] {meaning}" in rules
