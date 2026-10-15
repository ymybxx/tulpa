import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fish_plugin_sdk import Event, Plugin, ReviewCriterion, ReviewProfile
from pydantic import BaseModel

from fish_memory.extensions.reviews import PluginReviews, evidence_window

PROFILE = ReviewProfile(
    title="运行建议",
    goal="有自己的想法，并根据真实反馈接续交流。",
    criteria=[ReviewCriterion(name="重复", description="区分话题接续和同句重复")],
)
BASE = datetime(2026, 9, 11, 0, 0, tzinfo=UTC)


def sample(index, *, status="completed", text="想种一盆薄荷。", offset=None):
    at = BASE + timedelta(seconds=20 * index)
    return {
        "event_id": str(uuid4()),
        "type": "live.idle",
        "status": status,
        "disposition": "responding",
        "data": {"connection": "not_configured"},
        "text": text,
        "consumed_offset": len(text) if offset is None else offset,
        "received_at": at.isoformat(),
        "finished_at": (at + timedelta(seconds=5)).isoformat(),
        "model": "test-chat",
        "generation_us": 500_000,
    }


def window(count=6):
    return {
        "events": [sample(i) for i in range(count)],
        "conversation_messages": [],
        "total_events": count,
    }


class Reviewer:
    def __init__(self, *, bad_id=False, fail=False):
        self.calls = []
        self.bad_id, self.fail = bad_id, fail
        self.entered = asyncio.Event()
        self.release = None

    async def stream_json(self, messages, *, response_schema=None):
        self.calls.append((messages, response_schema))
        self.entered.set()
        if self.release:
            await self.release.wait()
        if self.fail:
            raise ValueError("invalid provider response")
        value = {
            "turns": [
                {
                    "evidence_id": e["id"],
                    "material": "同一话题",
                    "grounding": "not_applicable",
                    "grounding_reason": "这是个人想法，没有外部事实前提。",
                    "contribution": e["text"],
                    "connection": "specific",
                    "depends_on_reply": False,
                    "repeats": [],
                }
                for e in json.loads(messages[-1]["content"])["evidence"]
                if e["source"] == "plugin" and e.get("text")
            ],
            "summary": "话题可以保留，但连续发言缺少推进。",
            "strengths": [],
            "findings": [
                {
                    "title": "同句重复",
                    "criterion": "重复",
                    "severity": "major",
                    "target": "plugin_logic",
                    "evidence_ids": ["invented" if self.bad_id else "E1", "E2"],
                    "observation": "两个已播片段完全相同。",
                    "suggestion": "检查历史上下文，再复测同主题的连续发言。",
                    "confidence": "high",
                }
            ],
            "limitations": ["没有真实观众回应，不能判断观众偏好。"],
        }
        yield SimpleNamespace(content=json.dumps(value, ensure_ascii=False))


async def setup(
    tmp_path, *, count=6, model=None, foreground=lambda: False, interval=180
):
    store = SimpleNamespace(review_window=AsyncMock(return_value=window(count)))
    model = model or Reviewer()
    reviews = PluginReviews(
        root=tmp_path,
        store=store,
        llm=model,
        fishes=SimpleNamespace(
            get_profile=AsyncMock(
                return_value=SimpleNamespace(name="鱼", role="宠物", setting="好奇")
            )
        ),
        foreground=foreground,
        interval_seconds=interval,
    )
    instance = SimpleNamespace(
        id=uuid4(),
        fish_id=uuid4(),
        session_id=uuid4(),
        generation=uuid4(),
        plugin_id="test.live",
    )
    await reviews.start(instance, "1.2.3", PROFILE.model_dump())
    return reviews, instance, model, store


def test_metrics_only_count_device_confirmed_prefixes():
    a = sample(0, text="你好。未播尾巴", offset=3)
    b = sample(1, status="generated", text="尚未播出的问题？")
    c = sample(2, status="interrupted", text="", offset=0)
    values, metrics = evidence_window({"events": [a, b, c]})
    assert values[0]["text"] == "你好。"
    assert values[1]["text"] == values[2]["text"] == ""
    assert metrics["spoken_responses"] == 1
    assert metrics["question_endings"] == 0
    assert metrics["after_playback_to_request_median_seconds"] == 15


def test_merged_event_receipts_count_one_spoken_response():
    first, second = (
        sample(0, text="欢迎两位。", offset=5),
        sample(1, text="欢迎两位。", offset=5),
    )
    second["message_id"] = first["message_id"] = str(uuid4())
    ids = [first["event_id"], second["event_id"]]
    first["response_event_ids"] = second["response_event_ids"] = ids
    values, metrics = evidence_window({"events": [first, second]})
    assert metrics["spoken_responses"] == 1
    assert metrics["exact_duplicate_responses"] == 0
    assert values[1]["shared_response"] and not values[1]["text"]
    assert values[1]["response_event_ids"] == ids


def test_review_keeps_actual_context_audit_without_claiming_old_records_have_it():
    event = sample(0)
    event["context_audit"] = {
        "activity": {"role": "主播"},
        "history_messages": 179,
        "latest_user_input": "你来主持",
        "history_format": "quoted_records",
    }
    evidence, _ = evidence_window({"events": [event, sample(1)]})
    assert evidence[0]["context_audit"] == event["context_audit"]
    assert evidence[1]["context_audit"] is None


def test_review_distinguishes_planned_content_from_actual_played_prefix():
    event = sample(0, status="interrupted", text="这剧情。剩下的分析没播", offset=4)
    event["plan"] = {"point": "讲清楚角色选择背后的问题"}
    event["perception"] = {
        "available": True,
        "text": "旧游戏资料" * 1000 + "\n\n最新剧情是主动离婚。",
    }
    evidence, metrics = evidence_window({"events": [event]})
    assert evidence[0]["text"] == "这剧情。"
    assert evidence[0]["plan"] == event["plan"]
    assert "最新剧情" in evidence[0]["perception"]["text"]
    assert metrics["spoken_chars"] == 4


def test_regular_chat_reply_is_context_without_inflating_plugin_speech():
    user_id = str(uuid4())
    messages = [
        {
            "message_id": user_id,
            "role": "user",
            "text": "不看了",
            "at": BASE.isoformat(),
        },
        {
            "message_id": str(uuid4()),
            "role": "assistant",
            "text": "好。没播的尾巴",
            "status": "completed",
            "consumed_offset": 2,
            "reply_to_message_id": user_id,
            "at": BASE.isoformat(),
        },
    ]
    values, metrics = evidence_window({"events": [], "conversation_messages": messages})
    assert values[1]["source"] == "assistant"
    assert values[1]["text"] == "好。"
    assert values[1]["reply_to_message_id"] == user_id
    assert metrics["spoken_responses"] == 0
    messages[1]["consumed_offset"] = None
    assert (
        evidence_window({"events": [], "conversation_messages": messages})[0][1][
            "delivery"
        ]
        == "visible_text"
    )
    messages[1]["status"] = "generating"
    pending = evidence_window({"events": [], "conversation_messages": messages})[0][1]
    assert not pending["text"] and pending["delivery"] == "awaiting_delivery"


@pytest.mark.asyncio
async def test_automatic_assessment_persists_scoped_evidence_and_survives_restart(
    tmp_path,
    monkeypatch,
):
    read_text = Path.read_text

    def default_gbk(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "gbk", errors=errors)

    monkeypatch.setattr(Path, "read_text", default_gbk)
    reviews, instance, model, store = await setup(tmp_path)
    frame = {
        "width": 960,
        "height": 623,
        "byteCount": 125018,
        "digest": "f" * 64,
        "observedAt": BASE.isoformat(),
    }
    store.review_window.return_value["events"][0]["screen_frame"] = frame
    try:
        run = reviews.runs[instance.id]
        await reviews._analyse(run, manual=False)
        page = await reviews.history(instance.id)
        report = page["items"][0]
        assert report["trigger"] == "automatic" and report["status"] == "completed"
        assert report["run_id"] == str(instance.generation)
        assert report["plugin_version"] == "1.2.3"
        assert report["evidence"][0]["text"] == "想种一盆薄荷。"
        assert report["evidence"][0]["screen_frame"] == frame
        assert report["metrics"]["exact_duplicate_responses"] == 5
        assert store.review_window.call_args.args[:4] == (
            instance.id,
            instance.fish_id,
            instance.session_id,
            instance.generation,
        )
        request = json.loads(model.calls[0][0][-1]["content"])
        assert request["current_personality"]["name"] == "鱼"
        assert request["evidence"][0]["screen_frame"] == {
            "attached": True,
            "observed_at": BASE.isoformat(),
        }
        assert "screen_frame" not in request["evidence"][1]
        assert all(
            field not in model.calls[0][0][-1]["content"]
            for field in ("width", "height", "byteCount", frame["digest"])
        )
        assert model.calls[0][1]["$defs"]["Finding"]["properties"]["evidence_ids"][
            "items"
        ]["enum"] == ["E1", "E2", "E3", "E4", "E5", "E6"]
        await reviews.stop(instance.id)
        restored = PluginReviews(
            root=tmp_path,
            store=store,
            llm=model,
            fishes=reviews.fishes,
            foreground=lambda: False,
        )
        assert restored.state(instance.id)["phase"] == "stopped"
        assert (await restored.history(instance.id))["items"][0] == report
        assert len(model.calls) == 1
        assert report["analysis_metrics"]["input_chars"] > 0
        assert report["analysis_metrics"]["output_chars"] > 0
        assert report["analysis_metrics"]["first_content_ms"] is not None
        assert report["analysis_metrics"]["duration_ms"] >= 0
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_failed_review_preserves_partial_response_timing(tmp_path):
    class PartialFailure:
        async def stream_json(self, *args, **kwargs):
            yield SimpleNamespace(content='{"turns":[')
            raise TimeoutError

    reviews, instance, _, _ = await setup(tmp_path, model=PartialFailure())
    try:
        run = reviews.runs[instance.id]
        before = run.cursor
        await reviews._analyse(run, manual=True)
        report = (await reviews.history(instance.id))["items"][0]
        assert report["status"] == "failed" and run.cursor == before
        assert report["analysis_metrics"]["output_chars"] == len('{"turns":[')
        assert report["analysis_metrics"]["first_content_ms"] is not None
        assert report["analysis_metrics"]["duration_ms"] >= 0
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_backlog_report_names_the_sampled_window_and_omissions(tmp_path):
    reviews, instance, _, store = await setup(tmp_path)
    store.review_window.return_value["total_events"] = 100
    try:
        run = reviews.runs[instance.id]
        before = run.cursor
        await reviews._analyse(run, manual=True)
        report = (await reviews.history(instance.id))["items"][0]
        assert report["omitted_earlier_events"] == 94
        assert report["requested_after"] == before.isoformat(timespec="seconds")
        assert (
            report["started_at"]
            == store.review_window.return_value["events"][0]["received_at"]
        )
        assert run.cursor > before
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_full_review_keeps_early_feedback_when_context_is_large(tmp_path):
    reviews, instance, model, store = await setup(tmp_path, count=15)
    data = window(15)
    for e in data["events"]:
        e["perception"] = {"available": True, "text": "相同的近期影音材料。" * 500}
    data["conversation_messages"] = [
        {
            "message_id": str(uuid4()),
            "role": "user",
            "text": "可以收回刚才的结论。",
            "at": BASE.isoformat(),
        }
    ]
    store.review_window.return_value = data
    try:
        await reviews._analyse(reviews.runs[instance.id], manual=True)
        report = (await reviews.history(instance.id))["items"][0]
        assert report["status"] == "completed"
        assert len(report["assessment"]["turns"]) == 15
        assert report["metrics"]["analysed_spoken_responses"] == 15
        request = json.loads(model.calls[0][0][-1]["content"])
        assert "U1" in {e["id"] for e in request["evidence"]}
        assert request["omitted_evidence_count"] == 0
        assert request["contexts"]
        assert len(model.calls) == 1
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_insufficient_sample_does_not_call_model_or_spam_history(tmp_path):
    reviews, instance, model, _ = await setup(tmp_path, count=1)
    try:
        run = reviews.runs[instance.id]
        await reviews._analyse(run, manual=False)
        await reviews._analyse(run, manual=False)
        assert not model.calls and not (await reviews.history(instance.id))["items"]
        await reviews._analyse(run, manual=True)
        report = (await reviews.history(instance.id))["items"][0]
        assert report["status"] == "insufficient" and not model.calls
        assert run.cursor == run.started_at
    finally:
        await reviews.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"bad_id": True}, {"fail": True}])
async def test_bad_review_is_recorded_as_failure_without_advancing_or_applying(
    tmp_path, options
):
    reviews, instance, model, _ = await setup(tmp_path, model=Reviewer(**options))
    try:
        run = reviews.runs[instance.id]
        cursor = run.cursor
        await reviews._analyse(run, manual=True)
        report = (await reviews.history(instance.id))["items"][0]
        assert report["status"] == "failed" and "assessment" not in report
        assert run.running and run.cursor == cursor
        assert set(p.parent.name for p in tmp_path.rglob("*.json")) == {
            "runs",
            str(instance.id),
        }
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_request_is_nonblocking_and_stop_cancels_reviewer(tmp_path):
    model = Reviewer()
    model.release = asyncio.Event()
    reviews, instance, _, _ = await setup(tmp_path, model=model)
    try:
        assert reviews.request(instance.id)["phase"] == "queued"
        await asyncio.wait_for(model.entered.wait(), 1)
        assert reviews.request(instance.id)["phase"] == "analysing"
        assert len(model.calls) == 1
        await reviews.stop(instance.id)
        assert not (await reviews.history(instance.id))["items"]
        assert reviews.state(instance.id)["phase"] == "stopped"
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_foreground_defers_review_and_new_run_never_reuses_old_binding(tmp_path):
    reviews, instance, model, store = await setup(tmp_path, foreground=lambda: True)
    try:
        await reviews._analyse(reviews.runs[instance.id], manual=True)
        assert reviews.state(instance.id)["phase"] == "waiting_user"
        assert reviews.runs[instance.id].manual
        assert not model.calls and not store.review_window.called
        instance.fish_id, instance.session_id, instance.generation = (
            uuid4(),
            uuid4(),
            uuid4(),
        )
        await reviews.start(instance, "2.0.0", PROFILE.model_dump())
        reviews.foreground = lambda: False
        await reviews._analyse(reviews.runs[instance.id], manual=True)
        assert store.review_window.call_args.args[:4] == (
            instance.id,
            instance.fish_id,
            instance.session_id,
            instance.generation,
        )
        assert (await reviews.history(instance.id))["items"][0][
            "plugin_version"
        ] == "2.0.0"
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_history_pages_and_rejects_paths_outside_history(tmp_path):
    reviews, instance, _, _ = await setup(tmp_path)
    try:
        for index in range(12):
            await reviews._persist(
                tmp_path
                / "reports"
                / str(instance.id)
                / f"20260911T{index:012d}-{uuid4()}.json",
                {"order": index},
            )
        first = await reviews.history(instance.id)
        second = await reviews.history(instance.id, first["next_cursor"])
        assert [r["order"] for r in first["items"]] == list(range(11, 1, -1))
        assert [r["order"] for r in second["items"]] == [1, 0]
        with pytest.raises(ValueError):
            await reviews.history(instance.id, "../../secrets.json")
    finally:
        await reviews.close()


@pytest.mark.asyncio
async def test_manual_review_of_stopped_run_does_not_claim_to_be_running(tmp_path):
    reviews, instance, _, store = await setup(tmp_path)
    try:
        await reviews.stop(instance.id)
        await reviews._analyse(reviews.runs[instance.id], manual=True)
        assert (
            store.review_window.call_args.args[-1]
            == reviews.runs[instance.id].stopped_at
        )
        assert (await reviews.history(instance.id))["items"][0]["status"] == "completed"
        assert reviews.state(instance.id)["phase"] == "stopped"
        assert not reviews.state(instance.id)["running"]
    finally:
        await reviews.close()


def test_review_profile_is_metadata_and_does_not_change_event_definition():
    class Data(BaseModel):
        text: str

    event = Event("test.event", Data, "原来的发言描述", True)

    class Example(Plugin):
        events = (event,)

    original = Example.describe()
    assert "review" not in original
    Example.review = PROFILE
    assert Example.describe()["events"] == original["events"]
    assert Example.describe()["review"]["title"] == "运行建议"


@pytest.mark.asyncio
async def test_background_timer_produces_report_without_manual_request(
    tmp_path, monkeypatch
):
    reviews, instance, model, store = await setup(tmp_path, interval=0.03)
    store.review_window.side_effect = [window(6)] + [window(0)] * 20
    saved = asyncio.Event()
    original = reviews._save_run

    async def notify_saved(run):
        await original(run)
        if run.last_review_at is not None:
            saved.set()

    monkeypatch.setattr(reviews, "_save_run", notify_saved)
    try:
        await asyncio.wait_for(saved.wait(), 2)
        assert len(model.calls) == 1
        assert (await reviews.history(instance.id))["items"][0][
            "trigger"
        ] == "automatic"
    finally:
        await reviews.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped_manual", [False, True])
async def test_deferred_review_resumes_when_foreground_finishes(
    tmp_path, stopped_manual
):
    busy = True
    checked = asyncio.Event()

    def foreground():
        checked.set()
        return busy

    reviews, instance, model, _ = await setup(
        tmp_path, interval=0.03, foreground=foreground
    )
    try:
        if stopped_manual:
            await reviews.stop(instance.id)
            reviews.request(instance.id)
        await asyncio.wait_for(checked.wait(), 1)
        assert not model.calls
        busy = False
        await asyncio.wait_for(model.entered.wait(), 1)
        assert len(model.calls) == 1
    finally:
        await reviews.close()
