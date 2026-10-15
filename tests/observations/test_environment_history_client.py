from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from PIL import Image
from pydantic import ValidationError

import fish_memory.observations.environment_history_audio as audio_module
from fish_memory.clock import format_prompt_time_range, localize_passage_timestamps
from fish_memory.companion.shared_asr import (
    SHARED_ASR_SOCKET_ENV,
    SharedAsrResponse,
)
from fish_memory.context_descriptions import AUDIOVISUAL_SUMMARY_RULES
from fish_memory.observations.environment_history_audio import (
    SystemAudioSegment,
    SystemAudioTranscript,
)
from fish_memory.observations.environment_history_client import (
    EnvironmentHistoryModelClient,
    EnvironmentHistoryResponseError,
)
from fish_memory.observations.environment_history_contact_sheet import (
    build_contact_sheet,
)
from fish_memory.observations.environment_history_models import (
    EnvironmentHistoryWindow,
    SemanticDocument,
    SemanticSnapshot,
)
from fish_memory.observations.environment_history_store import EnvironmentHistoryStore

FISH = UUID(int=1)
CAPTURE = UUID(int=2)
START = datetime(2026, 9, 3, tzinfo=UTC)


def _jpeg(color: str) -> bytes:
    stream = BytesIO()
    with Image.new("RGB", (2, 1), color) as image:
        image.save(stream, format="JPEG")
    return stream.getvalue()


JPEG = _jpeg("red")
JPEG_2 = _jpeg("blue")
PCM = b"\x01\x00" * 160_000
AUDIBLE_PCM_1 = b"\x01\x01"
AUDIBLE_PCM_2 = b"\x02\x02"


def _shared_socket_path() -> Path:
    return Path("/private/tmp") / f"fma-{uuid4().hex}.sock"


def _window() -> EnvironmentHistoryWindow:
    return EnvironmentHistoryWindow.model_validate(
        {
            "captureSessionId": str(CAPTURE),
            "windowIndex": 0,
            "startedAt": START.timestamp(),
            "endedAt": START.timestamp() + 10,
            "frames": [
                {
                    "frameId": str(UUID(int=index + 10)),
                    "revision": index + 1,
                    "observedAt": START.timestamp() + offset,
                    "expiresAt": START.timestamp() + offset + 3,
                    "displayID": 1,
                    "pixelWidth": 2,
                    "pixelHeight": 1,
                    "byteLength": len((JPEG, JPEG_2)[index]),
                    "mimeType": "image/jpeg",
                    "jpegBase64": base64.b64encode((JPEG, JPEG_2)[index]).decode(),
                }
                for index, offset in enumerate((1, 9))
            ],
            "audioStatus": "complete",
            "audioPcmBase64": base64.b64encode(PCM).decode(),
            "audioSampleRate": 16000,
            "audioCapturedSamples": 160_000,
        }
    )


def _snapshot(index: int, *, period_seconds: int = 10) -> SemanticSnapshot:
    return SemanticSnapshot(
        id=UUID(int=100 + index),
        fish_id=FISH,
        capture_session_id=CAPTURE,
        window_index=index,
        period_seconds=period_seconds,
        started_at=START + timedelta(seconds=index * period_seconds),
        ended_at=START + timedelta(seconds=(index + 1) * period_seconds),
        document=SemanticDocument(
            summary=f"观察片段 {index}", audio=f"本地转写 {index}"
        ),
        input_digest=f"{index:064d}",
        created_at=START + timedelta(minutes=2),
    )


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.finish_reason = "stop"
        self.content = _response(
            "奶奶借给孙子一本书，约定下周归还。", "用户在编辑器修改代码。"
        )
        self.closed = False

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )

    async def close(self) -> None:
        self.closed = True


class SequencedModel(FakeModel):
    def __init__(self, responses: list[tuple[str, str] | Exception]) -> None:
        super().__init__()
        self.responses = responses

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content, finish_reason = response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def _client(model: FakeModel) -> EnvironmentHistoryModelClient:
    return EnvironmentHistoryModelClient(
        api_key="unused",
        base_url="https://unused.invalid/v1",
        model="qwen3.7-flash",
        client=cast(Any, model),
    )


def _response(audiovisual: str, activity: str = "") -> str:
    return json.dumps(
        {"audiovisualSummary": audiovisual, "activitySummary": activity},
        ensure_ascii=False,
    )


def _categorized_snapshot(index: int, *, period_seconds: int = 10) -> SemanticSnapshot:
    return replace(
        _snapshot(index, period_seconds=period_seconds),
        document=SemanticDocument(
            audiovisual_summary=f"奶奶与孙子讨论借书，第 {index} 段。",
            activity_summary=f"用户编辑代码，第 {index} 段。",
            audio=f"对白 {index}：下周一定归还。",
        ),
    )


def _time_range(snapshot: SemanticSnapshot) -> str:
    return (
        f"{snapshot.started_at:%Y-%m-%d %H:%M:%S} — "
        f"{snapshot.ended_at:%Y-%m-%d %H:%M:%S} UTC"
    )


def _minute_audio(snapshots: tuple[SemanticSnapshot, ...]) -> str:
    return "\n\n".join(
        f"{_time_range(item)}\n{audio}"
        for item in snapshots
        if (audio := audio_module.format_system_audio_text(item.document.audio).strip())
    )


class EventModel(FakeModel):
    """Fake narrative service selecting actual input sources, without semantics."""

    async def create(self, **kwargs):
        if '"required":["paragraphs"]' not in kwargs["messages"][0]["content"]:
            return await super().create(**kwargs)
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        paragraphs = [
            {
                "text": (
                    item.get("audio")
                    or item.get("mediaSummary")
                    or item.get("summary")
                    or "".join(t["words"] for t in item["turns"])
                )[:200],
                "sources": [item["sourceId"]],
            }
            for item in payload["segments"]
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"paragraphs": paragraphs}, ensure_ascii=False
                        )
                    ),
                )
            ]
        )


def _paragraph_payload(
    source: int = 1, text: str = "有人提出请求，尚未确认成功。"
) -> str:
    return json.dumps(
        {"paragraphs": [{"text": text, "sources": [source]}]}, ensure_ascii=False
    )


def _event_payload(source_id: str | int, **overrides) -> str:
    event = {
        "subject": "请求者",
        "action": "请求",
        "object": "对方",
        "status": "requested",
        "summary": "请求者提出请求，尚未确认成功。",
        "evidence": [{"sourceId": 1 if isinstance(source_id, str) else source_id}],
    }
    return json.dumps({"events": [{**event, **overrides}]}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_leaf_combines_words_and_contact_sheet_without_old_narrative():
    model = FakeModel()
    transcript = SystemAudioTranscript("奶奶，这本书借我读吧。", "Speech")
    previous = SemanticDocument(
        audiovisual_summary="旧的故事已经结束。", activity_summary="旧电脑活动"
    )
    result = await _client(model).summarize_window(
        _window(), transcript, previous_document=previous
    )
    assert result.audio == transcript.text
    assert result.audiovisual_summary == "奶奶借给孙子一本书，约定下周归还。"
    assert not result.visual_summary
    assert result.activity_summary == "用户在编辑器修改代码。"
    assert result.summary_version == 4
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["max_tokens"] == 3000
    assert call["extra_body"] == {"enable_thinking": False}
    assert call["messages"][0]["content"] == AUDIOVISUAL_SUMMARY_RULES
    assert call["response_format"] == {"type": "json_object"}
    content = call["messages"][1]["content"]
    assert content[0]["text"] == (
        "这是当前时段看到的画面：时间"
        + format_prompt_time_range(START, START + timedelta(seconds=10))
        + "\n你听到的声音："
        + transcript.text
    )
    assert all(
        word not in json.dumps(content, ensure_ascii=False)
        for word in (
            previous.audiovisual_summary,
            "previousDocument",
            "input_audio",
            "audioPcmBase64",
        )
    )
    assert [item["type"] for item in content] == ["text", "image_url"]
    await _client(model).close()
    assert model.closed


@pytest.mark.asyncio
async def test_audio_without_images_is_summarized_without_images_or_old_summary():
    model = FakeModel()
    model.content = _response("有人想借一本书。")
    result = await _client(model).summarize_window(
        _window().model_copy(update={"frames": ()}),
        SystemAudioTranscript("你好，我想借本书。", "Speech"),
        previous_document=SemanticDocument(summary="昨天已经还书。"),
    )
    assert result.audio == "你好，我想借本书。"
    assert result.audiovisual_summary == "有人想借一本书。"
    assert not result.activity_summary and not result.visual_summary
    assert len(model.calls) == 1
    content = model.calls[0]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text"]
    assert content[0]["text"] == (
        "当前时段：时间"
        + format_prompt_time_range(START, START + timedelta(seconds=10))
        + "\n你听到的声音：你好，我想借本书。"
    )
    assert "昨天已经还书" not in str(model.calls)


@pytest.mark.asyncio
async def test_ten_frames_use_one_request_and_keep_acoustic_words_unchanged():
    window = _window()
    frames = tuple(
        window.frames[i % 2].model_copy(
            update={
                "frame_id": UUID(int=1000 + i),
                "observed_at": START.timestamp() + i,
                "expires_at": START.timestamp() + i + 3,
            }
        )
        for i in range(10)
    )
    window = window.model_copy(update={"frames": frames})
    digest = window.input_digest()
    words = "[00:02.00—00:03.00] 声音abc-T02：我还没上车。"
    model = FakeModel()
    result = await _client(model).summarize_window(
        window, SystemAudioTranscript(words, "Speech")
    )
    assert result.audio == words
    assert window.input_digest() == digest
    assert len(model.calls) == 1
    content = model.calls[0]["messages"][1]["content"]
    assert len(content) == 2
    assert "同一屏幕在不同时刻的连续帧" in model.calls[0]["messages"][0]["content"]
    assert "01–10是帧编号" in model.calls[0]["messages"][0]["content"]
    assert content[0]["text"].endswith("\n你听到的声音：" + words)
    sheet = build_contact_sheet(window)
    assert len(sheet.panels) == 10
    assert content[1]["image_url"]["url"].endswith(sheet.jpeg_base64)


@pytest.mark.asyncio
async def test_unreadable_image_preserves_audio_and_does_not_claim_joint_evidence():
    window = _window()
    frame = window.frames[0].model_copy(update={"pixel_width": 10})
    model = FakeModel()
    result = await _client(model).summarize_window(
        window.model_copy(update={"frames": (frame,)}),
        SystemAudioTranscript("请等一等。", "Speech"),
    )
    assert result.audio == "请等一等。"
    assert result.audiovisual_summary == "听到的话：\n请等一等。"
    assert result.summary_version == 3
    assert model.calls == []


@pytest.mark.asyncio
async def test_joint_evidence_survives_minute_and_five_minute_json_roundtrip():
    minutes = []
    for minute in range(5):
        leaves = tuple(
            replace(
                _snapshot(i),
                document=SemanticDocument(
                    summary_version=4,
                    audio=f"对白{i}：我还在等。",
                    audiovisual_summary=f"同期画面{i}：对方拿着包走向门外。",
                    activity_summary="电脑活动，不能进入剧情材料。",
                ),
            )
            for i in range(minute * 6, (minute + 1) * 6)
        )
        model = EventModel()
        document = await _client(model).merge_minute(leaves)
        # Sort keys to reproduce JSONB's loss of insertion order.
        document = SemanticDocument.model_validate_json(
            json.dumps(document.model_dump(), sort_keys=True, ensure_ascii=False)
        )
        assert len(document.narrative_sources) == 12
        minutes.append(
            replace(
                _snapshot(minute, period_seconds=60),
                id=UUID(int=2000 + minute),
                window_index=minute * 6,
                document=document,
            )
        )
    model = SequencedModel(
        [(_paragraph_payload(), "stop"), (_response("", "活动"), "stop")]
    )
    merged = await _client(model).merge_five_minutes(minutes)
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    words = [s for s in payload["segments"] if "audio" in s]
    media = [s for s in payload["segments"] if "mediaSummary" in s]
    assert len(words) == len(media) == 30
    assert [s["mediaSummary"] for s in media] == [
        f"同期画面{i}：对方拿着包走向门外。" for i in range(30)
    ]
    assert "电脑活动" not in json.dumps(payload, ensure_ascii=False)
    assert len(merged.narrative_sources) == 60
    assert all("同期画面" not in s["audio"] for s in words)


@pytest.mark.asyncio
async def test_silent_joint_visual_evidence_can_form_a_narrative_without_audio():
    leaves = tuple(
        replace(
            _snapshot(i),
            document=SemanticDocument(
                summary_version=4,
                audiovisual_summary=f"画面{i}：人物先递出包，再离开。",
            ),
        )
        for i in range(6)
    )
    model = EventModel()
    result = await _client(model).merge_minute(leaves)
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert all("mediaSummary" in s and "audio" not in s for s in payload["segments"])
    assert result.narrative_paragraphs


@pytest.mark.asyncio
async def test_silent_visual_story_stays_available_without_claiming_audio():
    model = FakeModel()
    result = await _client(model).summarize_window(
        _window(), SystemAudioTranscript("", "Music")
    )
    assert result.audiovisual_summary and result.summary_version == 4
    assert not result.audio and not result.visual_summary
    assert "影音内容" in result.summary


@pytest.mark.asyncio
async def test_event_marked_music_keeps_understood_words():
    transcript = audio_module._transcript("<Music>唱过这一首歌</Music>", "Music")
    result = await _client(FakeModel()).summarize_window(_window(), transcript)
    assert result.audio == "唱过这一首歌"
    assert "Music" not in result.audiovisual_summary


@pytest.mark.asyncio
async def test_current_audio_replaces_old_inference_and_activity_in_merge_input():
    model = EventModel()
    windows = tuple(_categorized_snapshot(index) for index in range(6))
    previous = SemanticDocument(
        audiovisual_summary="旧剧情：已经成功了。",
        activity_summary="旧活动：查看诊断。",
    )
    result = await _client(model).merge_snapshots(
        windows, parent_period_seconds=60, previous_document=previous
    )
    assert len(model.calls) == 2
    audio_payload = json.loads(model.calls[0]["messages"][1]["content"])
    encoded = json.dumps(audio_payload, ensure_ascii=False)
    assert (
        "previousDocument" not in audio_payload
        and "previousEvents" not in audio_payload
    )
    assert "旧剧情" not in encoded and "用户编辑" not in encoded
    assert "奶奶与孙子讨论借书" not in encoded
    assert all(item.document.audio in encoded for item in windows)
    visible_payload = json.loads(model.calls[1]["messages"][1]["content"])
    visible = json.dumps(visible_payload, ensure_ascii=False)
    assert "对白" not in visible
    assert "用户编辑" in visible
    assert result.narrative_paragraphs and result.summary_version == 3
    assert result.narrative_sources == {
        f"{item.id}:0": f"{_time_range(item)}\n{item.document.audio}"
        for item in windows
    }
    assert result.audio == ""


@pytest.mark.asyncio
async def test_whole_window_sources_keep_adjacent_turns_together():
    windows = list(_snapshot(i) for i in range(6))
    windows[0] = replace(
        windows[0],
        document=SemanticDocument(
            summary="旧推断",
            audio=(
                "[00:00.00—00:01.00] 声音abc123-01：我在等。\n"
                "[00:02.00—00:03.00] 声音abc123-02：我的已经到了。"
            ),
        ),
    )
    model = EventModel()
    result = await _client(model).merge_minute(windows)
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    sources = payload["segments"]
    assert len(sources) == 6
    assert [t["speaker"] for t in sources[0]["turns"]] == [
        "声音abc123-01",
        "声音abc123-02",
    ]
    assert [t["words"] for t in sources[0]["turns"]] == ["我在等。", "我的已经到了。"]
    assert all("timeRange" in source for source in sources)
    assert (
        result.narrative_sources[result.narrative_paragraphs[0].source_ids[0]]
        == _time_range(windows[0]) + "\n" + windows[0].document.audio
    )


@pytest.mark.asyncio
async def test_diagnostics_never_enter_leaf_visual_request_or_audio_event_material():
    transcript = SystemAudioTranscript(
        "谢谢。",
        "Speech",
        segments=(
            SystemAudioSegment(
                0.0,
                1.0,
                "谢谢。",
                None,
                False,
                local_speaker_id="private-local-id",
                reason="match_ambiguous",
                clean_seconds=0.8,
                best_similarity=0.7171,
            ),
        ),
    )
    model = EventModel()
    leaf = await _client(model).summarize_window(_window(), transcript)
    assert "谢谢。" in leaf.audio and "未确定" in leaf.audio
    windows = tuple(replace(_snapshot(i), document=leaf) for i in range(6))
    await _client(model).merge_minute(windows)
    material = json.dumps(
        [call["messages"][1] for call in model.calls], ensure_ascii=False
    )
    assert all(
        secret not in material
        for secret in ("private-local-id", "match_ambiguous", "0.7171")
    )


@pytest.mark.asyncio
async def test_prior_event_is_context_only_and_cannot_be_cited_as_current_fact():
    from fish_memory.observations.environment_history_narrative import parse_events

    event = parse_events(_event_payload("past-source"), {"past-source": "我要申请。"})
    previous = SemanticDocument(
        audiovisual_summary="提出申请。", narrative_events=event, summary_version=2
    )
    model = SequencedModel([(_paragraph_payload(9999), "stop")])
    windows = tuple(_snapshot(i) for i in range(6))
    result = await _client(model).merge_snapshots(
        windows, parent_period_seconds=60, previous_document=previous
    )
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["previousWords"] == ["我要申请。"]
    assert all(isinstance(item["sourceId"], int) for item in payload["segments"])
    assert not result.narrative_paragraphs and result.narrative_sources
    assert all(item.document.audio in result.audiovisual_summary for item in windows)
    assert "提出申请" not in result.audiovisual_summary
    assert len(model.calls) == 1  # Semantic errors do not cause evidence-free repair.


@pytest.mark.parametrize(
    "change",
    [
        {"evidence": [{"sourceId": "wrong"}]},
        {"evidence": [{"sourceId": "s1", "quote": "伪造或省略的原话"}]},
        {"status": "probably_done"},
        {"evidence": []},
        {"narration": ""},
    ],
)
def test_event_references_and_status_contract_reject_invalid_response(change):
    from fish_memory.observations.environment_history_narrative import parse_events

    with pytest.raises(ValueError):
        parse_events(_event_payload("s1", **change), {"s1": "请求尚未成功。"})


def test_event_quotes_are_copied_by_code_without_model_ellipsis_or_rewording():
    from fish_memory.observations.environment_history_narrative import (
        parse_events,
        render_events,
    )

    source = "[00:00.00—00:02.00] 声音abc123-01：我想试试，还没答应呢。"
    events = parse_events(_event_payload("s1", status="waiting"), {"s1": source})
    assert events[0].evidence[0].quote == source
    assert render_events(events).endswith("（等待中）")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        ("我订好桌了，还要等。", "临时关门了，预约取消。", "我想再预约原来那家店。"),
        ("可以借我一本书吗？", "暂时不能借给你。", "我已经把自己的书借给小林了。"),
        ("代码准备发布。", "审核失败，发布取消了。", "同事的版本上线了，我的还没有。"),
    ],
)
async def test_three_domains_preserve_new_progress_without_wrong_old_summary(scenario):
    model = EventModel()
    snapshots = tuple(
        replace(
            _snapshot(i),
            document=SemanticDocument(
                audiovisual_summary="全部成功，事情已经结束。",
                activity_summary="正在查看历史回复：全部成功。",
                audio=scenario[i % len(scenario)],
            ),
        )
        for i in range(6)
    )
    result = await _client(model).merge_minute(snapshots)
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert "全部成功" not in json.dumps(payload, ensure_ascii=False)
    assert all(words in result.audiovisual_summary for words in scenario)
    assert result.narrative_paragraphs
    assert all(
        any(quote.endswith(words) for words in scenario)
        for quote in result.narrative_sources.values()
    )


@pytest.mark.asyncio
async def test_recursive_events_keep_quotes_after_raw_audio_is_not_duplicated():
    client = _client(EventModel())
    leaves = tuple(_snapshot(i) for i in range(6))
    minute = await client.merge_minute(leaves)
    minutes = tuple(
        replace(_snapshot(i, period_seconds=60), document=minute) for i in range(5)
    )
    five = await client.merge_snapshots(minutes, parent_period_seconds=300)
    assert five.audio == ""
    assert five.narrative_paragraphs
    original_key, original_words = next(iter(five.narrative_sources.items()))
    # A larger period reads evidence, not independent prior summary guesses.
    from fish_memory.observations.environment_history_client import _narrative_content

    parents = tuple(
        replace(_snapshot(i, period_seconds=300), document=five) for i in range(6)
    )
    content, sources, _ = _narrative_content(parents, None)
    assert sources[original_key] == original_words
    assert all(
        "audio" in item and "events" not in item
        for item in json.loads(content)["segments"]
    )


@pytest.mark.asyncio
async def test_legacy_without_audio_is_retained_without_becoming_event_evidence():
    windows = tuple(
        replace(_snapshot(i), document=SemanticDocument(summary=f"旧混合记录{i}"))
        for i in range(6)
    )
    model = FakeModel()
    result = await _client(model).merge_minute(windows)
    assert model.calls == []
    assert not result.narrative_events and not result.audiovisual_summary
    assert all(
        item.document.legacy_summary in result.legacy_summary for item in windows
    )


@pytest.mark.asyncio
async def test_unverified_old_av_is_not_promoted_to_quote_supported_fact():
    windows = tuple(
        replace(
            _snapshot(i),
            document=SemanticDocument(audiovisual_summary=f"旧模型推断{i}"),
        )
        for i in range(6)
    )
    model = FakeModel()
    result = await _client(model).merge_minute(windows)
    assert model.calls == []
    assert "旧模型推断0" in result.legacy_summary
    assert not result.audiovisual_summary and not result.narrative_events


@pytest.mark.asyncio
async def test_invalid_events_keep_complete_words_but_visuals_still_independent():
    windows = tuple(_categorized_snapshot(i) for i in range(6))
    model = SequencedModel(
        [("{bad json", "stop"), (_response("画面里的动作", "用户查看页面"), "stop")]
    )
    result = await _client(model).merge_minute(windows)
    assert all(item.document.audio in result.audiovisual_summary for item in windows)
    assert "奶奶与孙子讨论" not in result.audiovisual_summary
    assert result.activity_summary == "用户查看页面"
    assert result.visual_summary == "画面里的动作"
    assert not result.narrative_paragraphs and result.narrative_sources
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_failed_visible_merge_keeps_whole_category_without_polluting_audio():
    windows = tuple(
        replace(
            _snapshot(i),
            document=SemanticDocument(
                activity_summary=f"活动{i}", visual_summary=f"画面{i}"
            ),
        )
        for i in range(6)
    )
    model = SequencedModel([("bad", "stop"), ("bad", "stop")])
    result = await _client(model).merge_minute(windows)
    assert all(
        f"活动{i}" in result.activity_summary and f"画面{i}" in result.visual_summary
        for i in range(6)
    )
    assert not result.audiovisual_summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [TimeoutError("network unavailable"), ConnectionError("network unavailable")],
)
async def test_network_failure_is_not_disguised_as_successful_audio_fallback(failure):
    model = SequencedModel([failure])
    with pytest.raises(type(failure)):
        await _client(model).merge_minute(tuple(_snapshot(i) for i in range(6)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses",
    [
        [("", "stop")],
        [("broken", "stop"), ("still broken", "stop")],
        [(_response("partial"), "length")],
    ],
)
async def test_invalid_visual_response_preserves_words_without_semantic_invention(
    responses,
):
    model = SequencedModel(responses)
    words = "无论图像模型失败与否，都保留这一句完整的话。"
    result = await _client(model).summarize_window(
        _window(), SystemAudioTranscript(words, "Speech")
    )
    assert result.audiovisual_summary == f"听到的话：\n{words}"
    assert result.audio == words
    assert result.activity_summary == result.visual_summary == ""


@pytest.mark.asyncio
async def test_visual_structure_repair_sees_response_only_and_never_replays_images():
    model = SequencedModel(
        [
            ('{"audiovisualSummary":"已有视觉动作"}', "stop"),
            (_response("已有视觉动作"), "stop"),
        ]
    )
    result = await _client(model).summarize_window(
        _window(), SystemAudioTranscript("原始对白", "Speech")
    )
    repair = model.calls[1]["messages"][1]["content"]
    assert json.loads(repair) == {
        "untrustedModelResponse": '{"audiovisualSummary":"已有视觉动作"}'
    }
    assert "原始对白" not in repair and "image_url" not in repair
    assert result.audiovisual_summary == "已有视觉动作"
    assert result.audio == "原始对白"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,finish,expected_calls",
    [
        ("", "stop", 1),
        ("{}", "stop", 2),
        ("not json", "stop", 2),
        (_response("incomplete"), "length", 1),
    ],
)
async def test_empty_invalid_and_truncated_model_output_are_not_accepted(
    content, finish, expected_calls
):
    model = FakeModel()
    model.content, model.finish_reason = content, finish
    with pytest.raises(ValueError):
        await _client(model).summarize_window(
            _window(), SystemAudioTranscript("", "silence")
        )
    assert len(model.calls) == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent,child,count", [(60, 10, 6), (300, 60, 5), (1800, 300, 6)]
)
async def test_layer_size_order_and_overlap_are_checked_before_model_calls(
    parent, child, count
):
    valid = tuple(_snapshot(i, period_seconds=child) for i in range(count))
    model = FakeModel()
    client = _client(model)
    for invalid in (
        valid[:-1],
        tuple(reversed(valid)),
        (replace(valid[0], period_seconds=10 if child != 10 else 60), *valid[1:]),
    ):
        with pytest.raises(ValueError):
            await client.merge_snapshots(invalid, parent_period_seconds=parent)
    assert model.calls == []


@pytest.mark.asyncio
async def test_oversize_required_audio_input_is_not_truncated_or_sent():
    windows = tuple(
        replace(
            _snapshot(i),
            document=SemanticDocument(summary="完整来源", audio="字" * 8000),
        )
        for i in range(6)
    )
    model = FakeModel()
    with pytest.raises(EnvironmentHistoryResponseError, match="byte budget"):
        await _client(model).merge_minute(windows)
    assert model.calls == []
    assert all(len(item.document.audio) == 8000 for item in windows)


@pytest.mark.asyncio
async def test_compressed_parent_rejects_lossy_fallback_and_retains_children():
    windows = tuple(
        replace(
            _snapshot(i, period_seconds=300),
            document=SemanticDocument(summary="旧记录" * 80),
        )
        for i in range(6)
    )
    with pytest.raises(EnvironmentHistoryResponseError, match="retain children"):
        await _client(FakeModel()).merge_snapshots(windows, parent_period_seconds=1800)
    assert all(item.document.legacy_summary == "旧记录" * 80 for item in windows)


def test_event_and_visual_fields_round_trip_without_changing_legacy_serialization():
    from fish_memory.observations.environment_history_narrative import parse_events

    events = parse_events(_event_payload("s1"), {"s1": "我想试试。"})
    document = SemanticDocument(
        audiovisual_summary="有人提出尝试。",
        visual_summary="有人站在门口。",
        narrative_events=events,
        summary_version=2,
    )
    encoded = document.model_dump_json()
    assert SemanticDocument.model_validate_json(encoded) == document
    assert '"quote":"我想试试。"' in encoded
    assert SemanticDocument(summary="旧记录", audio="旧原话").model_dump() == {
        "summary": "旧记录",
        "audio": "旧原话",
    }
    with pytest.raises(ValidationError):
        SemanticDocument.model_validate({"summary": "x", "unexpected": True})


def test_omitted_current_turns_are_kept_as_unresolved_quotes_without_new_claims():
    from fish_memory.observations.environment_history_narrative import (
        parse_events,
        preserve_unmentioned_sources,
    )

    sources = {"s1": "已经预约，正在等待。", "s2": "店临时关门，取消了。"}
    events = preserve_unmentioned_sources(
        parse_events(_event_payload("s1"), sources), sources
    )
    assert len(events) == 2
    assert events[1].status == "uncertain"
    assert events[1].evidence[0].quote == sources["s2"]
    assert sources["s2"] in events[1].narration
    assert {e.source_id for event in events for e in event.evidence} == set(sources)


@pytest.mark.asyncio
async def test_five_minute_failure_retains_citable_sources_for_larger_parent():
    from fish_memory.observations.environment_history_client import _narrative_content

    minutes = tuple(_snapshot(i, period_seconds=60) for i in range(5))
    model = SequencedModel([("invalid", "stop")])
    five = await _client(model).merge_snapshots(minutes, parent_period_seconds=300)
    assert not five.audio
    assert five.narrative_sources and not five.narrative_paragraphs
    parents = tuple(
        replace(_snapshot(i, period_seconds=300), document=five) for i in range(6)
    )
    content, sources, legacy = _narrative_content(parents, None)
    assert content and not legacy
    assert all(
        any(words.endswith(item.document.audio) for words in sources.values())
        for item in minutes
    )
    assert not five.narrative_events


def test_unmentioned_sources_cannot_jump_over_intervening_state_change():
    from fish_memory.observations.environment_history_narrative import (
        parse_events,
        preserve_unmentioned_sources,
        render_events,
    )

    sources = {
        "first": "先提出请求。",
        "middle": "对方说取消了。",
        "last": "后来重新申请。",
    }
    events = parse_events(
        _event_payload(2, status="cancelled", summary="对方说取消了。"), sources
    )
    completed = preserve_unmentioned_sources(events, sources)
    text = render_events(completed)
    assert (
        text.index(sources["first"])
        < text.index(sources["middle"])
        < text.index(sources["last"])
    )
    assert len(completed) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["silence", "Music", "Speech", "unknown"])
async def test_processed_audio_without_words_is_a_valid_empty_child(event):
    client = _client(EventModel())
    window = _window().model_copy(update={"frames": ()})
    silent = await client.summarize_window(window, SystemAudioTranscript("", event))
    assert silent.summary_version == 3 and silent.summary == ""
    leaves = tuple(replace(_snapshot(i), document=silent) for i in range(6))
    minute = await client.merge_minute(leaves)
    assert minute.summary_version == 3 and not minute.summary
    assert client._client.calls == []
    with pytest.raises(ValidationError):
        SemanticDocument()


@pytest.mark.asyncio
async def test_silence_between_speech_does_not_block_parent_or_create_fake_content():
    client = _client(EventModel())
    leaves = tuple(
        _snapshot(i)
        if i in (0, 5)
        else replace(_snapshot(i), document=SemanticDocument(summary_version=2))
        for i in range(6)
    )
    result = await client.merge_minute(leaves)
    assert len(client._client.calls) == 1
    assert {words.splitlines()[-1] for words in result.narrative_sources.values()} == {
        "本地转写 0",
        "本地转写 5",
    }
    assert "无可用" not in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,event",
    [
        ("complete", "error"),
        ("complete", "unavailable"),
        ("partial", "silence"),
        ("unavailable", "silence"),
    ],
)
async def test_capture_or_transcription_failure_is_not_silently_accepted_as_silence(
    status, event
):
    window = _window().model_copy(update={"frames": (), "audio_status": status})
    with pytest.raises(EnvironmentHistoryResponseError, match="no interpretable"):
        await _client(FakeModel()).summarize_window(
            window, SystemAudioTranscript("", event)
        )


@pytest.mark.asyncio
async def test_large_parent_uses_same_event_contract_with_explicit_compression():
    from fish_memory.observations.environment_history_client import _event_system

    windows = tuple(_snapshot(i, period_seconds=300) for i in range(6))
    model = SequencedModel([(_paragraph_payload(1), "stop")])
    result = await _client(model).merge_snapshots(windows, parent_period_seconds=1800)
    system = model.calls[0]["messages"][0]["content"]
    assert system == _event_system(600, source_count=6)
    assert "600" in system and "paragraphs" in system
    assert len(result.audiovisual_summary) <= 600
    assert list(result.narrative_sources.values()) == [_time_range(windows[0])]
    request = json.loads(model.calls[0]["messages"][1]["content"])
    assert windows[0].document.audio in json.dumps(request, ensure_ascii=False)


def test_raw_turn_time_ranges_survive_minute_parent_without_reset_collision():
    from fish_memory.observations.environment_history_client import _narrative_content

    words = (
        "2026-09-03 00:00:00 — 2026-09-03 00:00:10 UTC\n"
        "[00:00.00—00:01.00] 声音abc123-T01：还在等。\n\n"
        "2026-09-03 00:00:10 — 2026-09-03 00:00:20 UTC\n"
        "[00:00.00—00:01.00] 声音abc123-T01：取消了。"
    )
    child = replace(
        _snapshot(0, period_seconds=60),
        document=SemanticDocument(summary="旧推断", audio=words),
    )
    content, _, _ = _narrative_content([child], None)
    segments = json.loads(content)["segments"]
    assert (
        segments[0]["timeRange"]
        == "2026-09-03T00:00:00+00:00 — 2026-09-03T00:00:10+00:00"
    )
    assert (
        segments[1]["timeRange"]
        == "2026-09-03T00:00:10+00:00 — 2026-09-03T00:00:20+00:00"
    )
    assert segments[0]["sourceId"] == 1 and segments[1]["sourceId"] == 2


@pytest.mark.asyncio
async def test_rich_six_five_minute_parents_can_compress_without_restoring_all_quotes():
    from fish_memory.observations.environment_history_narrative import parse_events

    windows = []
    for index in range(6):
        sources = {
            f"child-{index}-line-{j}": f"第{index}段第{j}项工作仍在等待确认。" * 8
            for j in range(10)
        }
        events = parse_events(
            json.dumps(
                {
                    "events": [
                        {
                            "summary": f"第{index}段第{j}项仍在等待确认。" * 4,
                            "status": "waiting",
                            "evidence": [{"sourceId": j + 1}],
                        }
                        for j in range(10)
                    ]
                },
                ensure_ascii=False,
            ),
            sources,
        )
        windows.append(
            replace(
                _snapshot(index, period_seconds=300),
                document=SemanticDocument(
                    audiovisual_summary="各项工作还在等待确认。" * 60,
                    narrative_events=events,
                    summary_version=2,
                ),
            )
        )
    answer = json.dumps(
        {
            "paragraphs": [
                {
                    "text": "六段工作均有待确认的事项，尚未全部完成。",
                    "sources": list(range(1, 7)),
                }
            ]
        },
        ensure_ascii=False,
    )
    model = SequencedModel([(answer, "stop")])
    document = await _client(model).merge_snapshots(windows, parent_period_seconds=1800)
    assert len(document.narrative_paragraphs) == 1
    assert len(document.narrative_paragraphs[0].source_ids) == 6
    assert len(document.narrative_sources) == 6
    assert len(document.audiovisual_summary) < 600
    assert len(document.model_dump_json().encode()) < 31_000
    assert all(len(item.document.narrative_events) == 10 for item in windows)
    request = json.loads(model.calls[0]["messages"][1]["content"])
    assert len(request["segments"]) == 6
    assert all(
        "summary" in segment and "audio" not in segment
        for segment in request["segments"]
    )


def test_loaded_sources_and_previous_words_follow_time_not_json_object_order():
    from fish_memory.observations.environment_history_client import _narrative_content

    sources = {
        "a:0": "2026-09-03 00:00:20 — 2026-09-03 00:00:30 UTC\n已经取消。",
        "m:0": "2026-09-03 00:00:10 — 2026-09-03 00:00:20 UTC\n正在等待。",
        "z:0": "2026-09-03 00:00:00 — 2026-09-03 00:00:10 UTC\n提出申请。",
    }
    previous = SemanticDocument(summary_version=3, narrative_sources=sources)
    child = replace(_snapshot(0, period_seconds=60), document=previous)
    content, ordered, _ = _narrative_content([child], previous)
    payload = json.loads(content)
    assert list(ordered) == ["z:0", "m:0", "a:0"]
    assert payload["previousWords"] == [
        localize_passage_timestamps(v) for v in ordered.values()
    ]
    assert [s["audio"].splitlines()[-1] for s in payload["segments"]] == [
        "提出申请。",
        "正在等待。",
        "已经取消。",
    ]


def test_leaf_words_cannot_replace_the_recorded_window_time_header():
    from fish_memory.observations.environment_history_client import _narrative_content

    date_like_words = "2099-01-01 00:00:00 — 2099-01-01 00:00:10 UTC\n提到了未来日期。"
    child = replace(
        _snapshot(0),
        document=SemanticDocument(summary_version=3, audio=date_like_words),
    )
    content, _, _ = _narrative_content([child], None)
    passage = json.loads(content)["segments"][0]
    assert passage["timeRange"] == localize_passage_timestamps(_time_range(child))
    assert date_like_words in passage["audio"]


def test_larger_period_reads_compressed_story_instead_of_empty_source_pointers():
    from fish_memory.observations.environment_history_client import _narrative_content
    from fish_memory.observations.environment_history_narrative import (
        NarrativeParagraph,
    )

    previous = SemanticDocument(
        summary_version=3,
        narrative_paragraphs=(
            NarrativeParagraph(
                text="申请已撤销，事情尚未办成。", source_ids=("old:summary",)
            ),
        ),
        narrative_sources={
            "old:summary": "2026-09-03 00:00:00 — 2026-09-03 00:05:00 UTC"
        },
    )
    child = replace(_snapshot(0, period_seconds=1800), document=previous)
    content, sources, _ = _narrative_content([child], previous, summary_sources=True)
    payload = json.loads(content)
    assert payload["segments"] == [
        {
            "sourceId": 1,
            "timeRange": localize_passage_timestamps(_time_range(child)),
            "summary": previous.audiovisual_summary,
        }
    ]
    assert payload["previousSummary"] == previous.audiovisual_summary
    assert "previousWords" not in payload
    assert list(sources) == [f"{child.id}:summary"]


def test_derived_event_narration_is_stored_once_and_round_trips_losslessly():
    from fish_memory.observations.environment_history_narrative import (
        parse_events,
        render_events,
    )

    events = parse_events(_event_payload(1), {"s1": "这次请求还没有成功。"})
    original = SemanticDocument(
        audiovisual_summary=render_events(events),
        narrative_events=events,
        summary_version=2,
        visual_summary="看到有人站在门口。",
        activity_summary="用户切换了页面。",
    )
    payload = original.model_dump()
    assert "audiovisualSummary" not in payload
    assert "subject" in payload["narrativeEvents"][0]  # Explicit legacy fields remain.
    restored = SemanticDocument.model_validate(payload)
    assert restored == original
    assert restored.summary == original.summary
    assert SemanticDocument.model_validate_json(original.model_dump_json()) == original
    different = original.model_copy(
        update={"audiovisual_summary": "独立但不同的旧描述。"}
    )
    assert different.model_dump()["audiovisualSummary"] == "独立但不同的旧描述。"


def test_derived_event_round_trip_survives_recursive_sensitive_text_redaction():
    from fish_memory.observations.environment_history_narrative import (
        parse_events,
        render_events,
    )
    from fish_memory.observations.environment_history_store import _redacted_document

    events = parse_events(
        _event_payload(1, summary="有人读出了 password=event-secret"),
        {"s1": "password=quote-secret"},
    )
    document = SemanticDocument(
        audiovisual_summary=render_events(events),
        narrative_events=events,
        summary_version=2,
    )
    assert "audiovisualSummary" not in document.model_dump()
    redacted = _redacted_document(document)
    assert "event-secret" not in redacted.summary
    assert "quote-secret" not in redacted.model_dump_json()
    assert redacted.audiovisual_summary == render_events(redacted.narrative_events)
    assert "[REDACTED]" in redacted.audiovisual_summary
    assert "audiovisualSummary" not in redacted.model_dump()
    assert redacted.narrative_events[0].evidence[0].source_id == "s1"


class RecordingPool:
    def __init__(self) -> None:
        self.parameters: tuple[object, ...] | None = None
        self.statement = ""

    async def execute(self, query: str, *parameters: object) -> str:
        self.statement = query
        self.parameters = parameters
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *parameters: object):
        del query
        assert self.parameters is not None
        p = self.parameters
        assert parameters[:2] == (FISH, p[0])
        return {
            "id": p[0],
            "fish_id": p[1],
            "payload": p[7],
            "started_at": p[8],
            "ended_at": p[9],
            "created_at": START,
        }


def _database_row(
    index: int, *, extra_document: dict[str, str] | None = None
) -> dict[str, object]:
    snapshot = _snapshot(index)
    document = snapshot.document.model_dump()
    document.update(extra_document or {})
    return {
        "id": snapshot.id,
        "fish_id": snapshot.fish_id,
        "payload": {
            "captureSessionId": str(snapshot.capture_session_id),
            "windowIndex": snapshot.window_index,
            "periodSeconds": snapshot.period_seconds,
            "inputDigest": snapshot.input_digest,
            "document": document,
            "documentMarkdown": "旧版持久化文本保持原样",
            "sourceSnapshotIds": [],
            "evidence": {},
        },
        "started_at": snapshot.started_at,
        "ended_at": snapshot.ended_at,
        "created_at": snapshot.created_at,
    }


class StoredRowsPool:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = {row["id"]: row for row in rows}

    async def fetchrow(self, query: str, *parameters: object):
        assert "source=$3" in query
        assert parameters[0] == FISH
        assert parameters[2] == "environment_history"
        return self.rows.get(parameters[1])


@pytest.mark.asyncio
async def test_legacy_database_uncertainties_are_read_but_not_resent_to_minute_model():
    rows = [
        _database_row(
            index,
            extra_document={
                "uncertainties": f"旧版内部不确定性说明 {index}",
            },
        )
        for index in range(6)
    ]
    store = EnvironmentHistoryStore(cast(Any, StoredRowsPool(rows)))
    loaded = [await store.get(FISH, UUID(int=100 + index)) for index in range(6)]
    snapshots = tuple(snapshot for snapshot in loaded if snapshot is not None)

    assert len(snapshots) == 6
    assert all(
        set(snapshot.document.model_dump()) == {"summary", "audio"}
        for snapshot in snapshots
    )

    model = FakeModel()
    client = _client(model)
    await client.merge_minute(snapshots)

    merge_payload = json.loads(model.calls[0]["messages"][1]["content"])
    serialized = json.dumps(merge_payload, ensure_ascii=False)
    assert "uncertainties" not in serialized
    assert "旧版内部不确定性说明" not in serialized
    await client.close()


@pytest.mark.asyncio
async def test_legacy_database_compatibility_does_not_accept_other_document_extras():
    row = _database_row(
        0,
        extra_document={
            "uncertainties": "允许丢弃的旧字段",
            "unexpected": "不能借兼容逻辑放行的字段",
        },
    )
    store = EnvironmentHistoryStore(cast(Any, StoredRowsPool([row])))

    with pytest.raises(ValidationError):
        await store.get(FISH, UUID(int=100))


@pytest.mark.asyncio
async def test_observation_storage_contains_semantics_and_metadata_only():
    pool = RecordingPool()
    store = EnvironmentHistoryStore(cast(Any, pool))
    snapshot = _snapshot(0)
    saved = await store.save(
        snapshot,
        evidence={
            "frameTimes": [START.timestamp() + 1, START.timestamp() + 9],
            "audioStatus": "complete",
            "audioCapturedSamples": 160_000,
        },
    )
    assert saved.document == snapshot.document
    assert pool.parameters is not None
    payload = json.loads(cast(str, pool.parameters[7]))
    assert payload["document"] == snapshot.document.model_dump()
    assert set(payload["document"]) == {"summary", "audio"}
    assert "documentMarkdown" not in payload
    assert "asrEvent" not in payload["evidence"]
    assert payload["evidence"]["audioCapturedSamples"] == 160_000
    assert "jpegBase64" not in payload and "audioPcmBase64" not in payload
    assert base64.b64encode(PCM).decode() not in json.dumps(payload)
    assert base64.b64encode(JPEG).decode() not in json.dumps(payload)
    assert base64.b64encode(JPEG_2).decode() not in json.dumps(payload)
    assert "deleted_at IS NULL" in pool.statement


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["accept", "finish"])
async def test_local_asr_failure_rebuilds_decoder_before_next_window(
    monkeypatch, failure_stage
):
    decoders = []

    class Decoder:
        def __init__(self, config):
            del config
            self.pcm = bytearray()
            self.first = not decoders
            decoders.append(self)

        def accept_pcm(self, pcm):
            self.pcm.extend(pcm)
            if self.first and failure_stage == "accept":
                raise RuntimeError("synthetic native accept failure")

        def finish_utterance_update(self):
            if self.first and failure_stage == "finish":
                raise RuntimeError("synthetic native decode failure")
            return SimpleNamespace(text="current window only", event="Speech")

    monkeypatch.setattr(audio_module, "StreamingAsr", Decoder)
    monkeypatch.setattr(
        audio_module,
        "AsrModelDownloader",
        lambda _: SimpleNamespace(ensure=lambda: None),
    )
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        with pytest.raises(RuntimeError):
            await transcriber.transcribe(AUDIBLE_PCM_1)
        transcript = await transcriber.transcribe(AUDIBLE_PCM_2)
        assert transcript.text == "current window only"
        assert len(decoders) == 2
        assert bytes(decoders[1].pcm) == AUDIBLE_PCM_2
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_local_asr_release_drops_decoder_then_lazily_reloads(monkeypatch):
    decoders = []

    class Decoder:
        def __init__(self, config):
            del config
            decoders.append(self)

        def accept_pcm(self, pcm):
            del pcm

        def finish_utterance_update(self):
            return SimpleNamespace(text="current window", event="Speech")

    monkeypatch.setattr(audio_module, "StreamingAsr", Decoder)
    monkeypatch.setattr(
        audio_module,
        "AsrModelDownloader",
        lambda _: SimpleNamespace(ensure=lambda: None),
    )
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        await transcriber.transcribe(AUDIBLE_PCM_1)
        assert len(decoders) == 1
        await transcriber.release()
        assert transcriber._decoder is None

        await transcriber.transcribe(AUDIBLE_PCM_2)
        assert len(decoders) == 2
        assert transcriber._decoder is decoders[1]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_queued_asr_release_cannot_drop_decoder_needed_by_new_work(monkeypatch):
    decoders = []

    class Decoder:
        def __init__(self, config):
            del config
            decoders.append(self)

        def accept_pcm(self, pcm):
            del pcm

        def finish_utterance_update(self):
            return SimpleNamespace(text="current window", event="Speech")

    monkeypatch.setattr(audio_module, "StreamingAsr", Decoder)
    monkeypatch.setattr(
        audio_module,
        "AsrModelDownloader",
        lambda _: SimpleNamespace(ensure=lambda: None),
    )
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    blocker_entered = threading.Event()
    unblock = threading.Event()
    try:
        await transcriber.transcribe(AUDIBLE_PCM_1)

        def block_worker() -> None:
            blocker_entered.set()
            assert unblock.wait(1)

        blocker = transcriber._executor.submit(block_worker)
        assert await asyncio.to_thread(blocker_entered.wait, 1)
        release = asyncio.create_task(transcriber.release())
        await asyncio.sleep(0)
        newer = asyncio.create_task(transcriber.transcribe(AUDIBLE_PCM_2))
        await asyncio.sleep(0)
        unblock.set()

        await asyncio.wait_for(asyncio.gather(release, newer), 1)
        blocker.result()
        assert len(decoders) == 1
        assert transcriber._decoder is decoders[0]
    finally:
        unblock.set()
        await transcriber.close()


@pytest.mark.asyncio
async def test_environment_audio_reuses_companion_asr_without_local_decoder(
    monkeypatch,
):
    class SharedClient:
        async def recognize(self, pcm):
            assert pcm == AUDIBLE_PCM_1
            return SharedAsrResponse("ok", "共享识别", "Speech")

    socket_path = _shared_socket_path()
    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(socket_path))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: SharedClient())
    monkeypatch.setattr(
        audio_module,
        "StreamingAsr",
        lambda _: (_ for _ in ()).throw(AssertionError("local ASR loaded")),
    )
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        transcript = await transcriber.transcribe(AUDIBLE_PCM_1)
        assert transcript == SystemAudioTranscript("共享识别", "Speech")
        assert transcriber._decoder is None
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_near_digital_silence_does_not_claim_shared_asr(monkeypatch):
    calls: list[bytes] = []

    class SharedClient:
        async def recognize(self, pcm):
            calls.append(pcm)
            return SharedAsrResponse("ok", "安静对白", "Speech")

    socket_path = _shared_socket_path()
    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(socket_path))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: SharedClient())
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        silent = await transcriber.transcribe(b"\x04\x00" * 160_000)
        audible = await transcriber.transcribe(b"\x05\x00" * 160_000)
        assert silent == SystemAudioTranscript("", "silence")
        assert audible == SystemAudioTranscript("安静对白", "Speech")
        assert calls == [b"\x05\x00" * 160_000]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_configured_shared_asr_unavailable_does_not_load_duplicate_model(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    socket_path = _shared_socket_path()
    monkeypatch.setattr(audio_module, "_SHARED_RECONNECT_SECONDS", 0)

    class SharedClient:
        async def recognize(self, pcm):
            del pcm
            return SharedAsrResponse("unavailable")

    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(socket_path))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: SharedClient())
    monkeypatch.setattr(
        audio_module,
        "StreamingAsr",
        lambda _: (_ for _ in ()).throw(AssertionError("local ASR loaded")),
    )
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        caplog.set_level("WARNING", logger=audio_module.__name__)
        transcript = await transcriber.transcribe(AUDIBLE_PCM_1)
        repeated = await transcriber.transcribe(AUDIBLE_PCM_2)
        assert transcript == SystemAudioTranscript(
            "", "unavailable", unavailable_reason="companion_recognizer_not_ready"
        )
        assert repeated == transcript
        assert transcriber._decoder is None
        assert "companion_recognizer_not_ready" in caplog.text
        assert "不加载第二份本地 ASR 模型" in caplog.text
        assert caplog.text.count("共享影音 ASR 本窗口不可用") == 1
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_concurrent_shared_environment_transcriptions_do_not_retire_each_other(
    monkeypatch,
):
    class SharedClient:
        def __init__(self):
            self.count = 0

        async def recognize(self, pcm):
            self.count += 1
            await asyncio.sleep(0)
            return SharedAsrResponse("ok", f"shared-{pcm[0]}", "Speech")

    socket_path = _shared_socket_path()
    shared_client = SharedClient()
    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(socket_path))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: shared_client)
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        first, second = await asyncio.gather(
            transcriber.transcribe(AUDIBLE_PCM_1),
            transcriber.transcribe(AUDIBLE_PCM_2),
        )
        assert {first.text, second.text} == {"shared-1", "shared-2"}
        assert shared_client.count == 2
        assert transcriber._decoder is None
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_voice_toggle_reconnects_same_media_pcm_without_a_second_model(
    monkeypatch,
):
    calls = []

    class SharedClient:
        async def recognize(self, pcm):
            calls.append(pcm)
            if len(calls) == 1:
                raise ConnectionRefusedError("old voice owner stopped")
            if len(calls) == 2:
                return SharedAsrResponse("unavailable")
            return SharedAsrResponse("ok", "切换期间的对白", "Speech")

    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(_shared_socket_path()))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: SharedClient())
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        result = await transcriber.transcribe(AUDIBLE_PCM_1)
        assert result.text == "切换期间的对白"
        assert calls == [AUDIBLE_PCM_1] * 3
        assert transcriber._decoder is None
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_media_socket_failure_has_a_bounded_retry_and_diagnostic(monkeypatch):
    calls = []

    class SharedClient:
        async def recognize(self, pcm):
            calls.append(pcm)
            raise ConnectionRefusedError("private path must not be exposed")

    monkeypatch.setenv(SHARED_ASR_SOCKET_ENV, str(_shared_socket_path()))
    monkeypatch.setattr(audio_module, "SharedAsrClient", lambda _: SharedClient())
    monkeypatch.setattr(audio_module, "_SHARED_RECONNECT_SECONDS", 0.02)
    transcriber = audio_module.LocalSystemAudioTranscriber(SimpleNamespace())
    try:
        result = await asyncio.wait_for(transcriber.transcribe(AUDIBLE_PCM_1), 1)
        assert result.event == "unavailable"
        assert result.unavailable_reason == "transport_ConnectionRefusedError"
        assert len(calls) >= 2
        assert transcriber._decoder is None
    finally:
        await transcriber.close()
