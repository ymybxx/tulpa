"""Deterministic timing and provider-correspondence tests; no billable calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.tts.facial import (
    FacialProfile,
    SpeechFacialPlan,
    create_speech_facial_plan,
)
from fish_memory.tts.routes import _forward_provider_events

PROFILE = FacialProfile(
    revision="model-v1",
    modelId="model",
    modelName="小猫",
    parameters=[
        {"id": "ParamAngleX", "minimum": -30, "maximum": 30, "defaultValue": 0}
    ],
)


async def drain():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.parametrize("wait_for_ready", [False, True])
async def test_ready_face_has_zero_wait_and_does_not_yield_to_event_loop(
    wait_for_ready,
):
    async def predict(text, previous):
        return {"ParamAngleX": 12}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=wait_for_ready)
    plan.append("你好。")
    await drain()
    yielded = []
    asyncio.get_running_loop().call_soon(yielded.append, True)
    cue = await plan.before_audio(0, "你好。")
    assert yielded == []  # A fixed cushion, sleep(0), or extra task would fail.
    assert cue["waitMs"] == 0
    assert cue["targets"] == {"ParamAngleX": 12}
    assert await plan.before_audio(0, "你好。") is None
    await plan.close()


async def test_only_the_remaining_late_face_time_is_waited(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("fish_memory.tts.facial.time.perf_counter", lambda: clock[0])
    done = asyncio.Event()

    async def predict(text, previous):
        await done.wait()
        return {"ParamAngleX": 0}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("真的呀！")
    await drain()
    gate = asyncio.create_task(plan.before_audio(0, "真的呀！"))
    await drain()
    assert not gate.done()
    clock[0] += 0.125
    done.set()
    cue = await gate
    assert cue["waitMs"] == 125.0
    await plan.close()


async def test_merged_sentences_and_repeated_text_after_rotation_are_distinct():
    calls = []

    async def predict(text, previous):
        calls.append((text, previous))
        return {"ParamAngleX": len(previous)}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("你好，")
    plan.append("又见面了。")
    await drain()
    plan.begin(0, "你好，又见面了。")
    cue = await plan.before_audio(0, "")  # Provider PCM annotations may be empty.
    assert cue["waitMs"] == 0
    assert cue["text"] == "你好，又见面了。"
    assert cue["targets"]["ParamAngleX"] == 0
    plan.rotate()
    plan.append("你好，又见面了。")
    cue = await plan.before_audio(0, "你好，又见面了。")
    assert cue["targets"]["ParamAngleX"] == len("你好，又见面了。")
    assert ("你好，又见面了。", "你好，又见面了。") in calls
    assert await plan.before_audio(1, "未提交过的文本。") is None
    await plan.close()


async def test_model_revision_change_drops_pending_result_and_stop_cancels_promptly():
    done = asyncio.Event()
    current = [True]

    async def predict(text, previous):
        await done.wait()
        return {"ParamAngleX": 1}

    plan = SpeechFacialPlan(
        PROFILE, predict, is_current=lambda: current[0], wait_for_ready=True
    )
    plan.append("旧模型。")
    gate = asyncio.create_task(plan.before_audio(0, "旧模型。"))
    await drain()
    current[0] = False
    done.set()
    assert await gate is None
    await plan.close()

    done.clear()
    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("被打断。")
    gate = asyncio.create_task(plan.before_audio(0, "被打断。"))
    await drain()
    plan.cancel_pending()
    assert await asyncio.wait_for(gate, 0.05) is None
    await plan.close()


async def test_error_and_timeout_release_audio_instead_of_leaving_it_stuck():
    async def predict(text, previous):
        await asyncio.Event().wait()

    plan = SpeechFacialPlan(PROFILE, predict, timeout=0.01, wait_for_ready=True)
    plan.append("继续播放。")
    cue = await plan.before_audio(0, "继续播放。")
    assert cue["targets"] is None and cue["error"] == "TimeoutError"
    await plan.close()


async def test_cloud_forwarder_sends_one_face_before_pcm_without_delaying_ready_audio():
    async def predict(text, previous):
        return {"ParamAngleX": 1}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("你好。")
    await drain()

    def event(kind, text=None):
        output = {"type": kind, "sentence": {"index": 0}}
        if text is not None:
            output["original_text"] = text
        return json.dumps(
            {"header": {"event": "result-generated"}, "payload": {"output": output}}
        )

    incoming = iter(
        [
            event("sentence-begin", "你好。"),
            event("sentence-synthesis"),
            b"\x00\x00",
            event("sentence-synthesis"),
            b"\x00\x00",
            json.dumps({"header": {"event": "task-finished"}}),
        ]
    )
    sent = []

    async def receive():
        return next(incoming)

    async def send(value):
        sent.append(value)

    await _forward_provider_events(
        SimpleNamespace(send_json=send, send_bytes=send),
        SimpleNamespace(receive=receive),
        started_at=0,
        timing={"text_sent_at": 0},
        stop_event=asyncio.Event(),
        facial=plan,
    )
    kinds = [item["type"] if isinstance(item, dict) else "pcm" for item in sent]
    assert kinds == ["sentence_begin", "facial", "audio", "pcm", "audio", "pcm"]
    audio = [
        item for item in sent if isinstance(item, dict) and item["type"] == "audio"
    ]
    assert audio[0]["facialWaitMs"] == 0
    assert audio[1]["facialWaitMs"] is None
    assert sent[1]["waitMs"] == 0
    await plan.close()


@pytest.mark.parametrize("first_wait_ms", [0, 700])
async def test_provider_first_packet_timing_excludes_face_wait_and_later_sentences(
    monkeypatch, first_wait_ms
):
    clock = [10.4]
    monkeypatch.setattr("fish_memory.tts.routes.time.perf_counter", lambda: clock[0])

    class Face:
        def set_sender(self, sender):
            pass

        async def before_audio(self, index, text):
            wait = first_wait_ms if index == 0 else 900
            clock[0] += wait / 1000
            return {"type": "facial", "waitMs": wait}

    def event(index):
        return json.dumps(
            {
                "header": {"event": "result-generated"},
                "payload": {
                    "output": {
                        "type": "sentence-synthesis",
                        "sentence": {"index": index},
                    }
                },
            }
        )

    incoming = iter(
        [
            event(0),
            b"\x00\x00",
            event(1),
            b"\x00\x00",
            json.dumps({"header": {"event": "task-finished"}}),
        ]
    )
    sent = []

    async def receive():
        return next(incoming)

    async def send(value):
        sent.append(value)

    first = await _forward_provider_events(
        SimpleNamespace(send_json=send, send_bytes=send),
        SimpleNamespace(receive=receive),
        started_at=10.0,
        timing={"text_sent_at": 10.0},
        stop_event=asyncio.Event(),
        facial=Face(),
    )
    audio = [
        item for item in sent if isinstance(item, dict) and item["type"] == "audio"
    ]
    assert first == 10.4
    assert audio[0]["firstPacketMs"] == 400
    assert audio[0]["facialWaitMs"] == first_wait_ms
    assert audio[1]["firstPacketMs"] is None
    assert audio[1]["facialWaitMs"] is None


async def test_shared_jev_works_without_enabling_experimental_behavior():
    calls = []

    async def evaluate(configuration, state, questions, *, kind):
        calls.append((configuration.enabled, state, kind))
        return {"answers": {"ParamAngleX": {"type": "score", "score": 1}}}

    state = SimpleNamespace(
        speech_facial_profile=PROFILE,
        experimental_companion=SimpleNamespace(
            configuration=ExperimentalConfiguration(enabled=False, api_key="fixture"),
            judge=SimpleNamespace(jev=SimpleNamespace(evaluate=evaluate)),
        ),
    )
    connection = SimpleNamespace(app=SimpleNamespace(state=state))
    assert create_speech_facial_plan(connection, False) is None
    plan = create_speech_facial_plan(connection, True)
    assert plan.wait_for_ready is False
    plan.append("你好。")
    await drain()
    assert (await plan.before_audio(0, "你好。"))["targets"] == {"ParamAngleX": 0}
    assert calls[0][0] is False and calls[0][2] == "speech.facial.jev"
    assert list(calls[0][1]) == ["previousSegments", "currentSegment"]
    await plan.close()


async def test_reused_provider_index_and_repeated_sentence_use_new_occurrence():
    async def predict(text, previous):
        return {"ParamAngleX": len(previous)}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("你好。你好。再见。")
    assert (await plan.before_audio(0, "你好。"))["targets"]["ParamAngleX"] == 0
    plan.end(0)
    assert (await plan.before_audio(0, "你好。"))["targets"]["ParamAngleX"] == 3
    plan.begin(0, "再见。")
    assert (await plan.before_audio(0, ""))["targets"]["ParamAngleX"] == 6
    await plan.close()


async def test_current_sentence_is_not_held_behind_queued_lookahead():
    calls = []
    block = asyncio.Event()

    async def predict(text, previous):
        calls.append(text)
        if text != "马上播放。":
            await block.wait()
        return {"ParamAngleX": 0}

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    for text in ["等待甲。", "等待乙。", "排队的未来。", "马上播放。"]:
        plan.append(text)
    await drain()
    assert calls == ["等待甲。", "等待乙。"]
    cue = await asyncio.wait_for(plan.before_audio(0, "马上播放。"), 0.1)
    assert cue["targets"] == {"ParamAngleX": 0}
    assert calls == ["等待甲。", "等待乙。", "马上播放。"]
    await plan.close()


async def test_audio_priority_bounds_inflight_jev_without_holding_audio():
    release = asyncio.Event()
    running = peak = 0

    async def predict(text, previous):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        try:
            await release.wait()
            return {"ParamAngleX": 0}
        finally:
            running -= 1

    plan = SpeechFacialPlan(PROFILE, predict)
    lines = [f"第{index}句。" for index in range(8)]
    try:
        for text in lines:
            plan.append(text)
        await drain()
        for index, text in enumerate(lines):
            assert await plan.before_audio(index, text) is None
            await drain()
        assert running == peak == 3  # Two lookahead slots plus one urgent slot.
        release.set()
        await drain()
    finally:
        await plan.close()
    assert running == 0 and peak == 3


async def test_audio_priority_does_not_yield_to_pending_jev_and_delivers_later():
    release = asyncio.Event()
    delivered = []

    async def predict(text, previous):
        await release.wait()
        return {"ParamAngleX": 12}

    async def send(cue):
        delivered.append(cue)

    plan = SpeechFacialPlan(PROFILE, predict)
    plan.set_sender(send)
    plan.append("你好。")
    await drain()
    yielded = []
    asyncio.get_running_loop().call_soon(yielded.append, True)
    assert await plan.before_audio(0, "你好。") is None
    assert yielded == []  # No wait for unfinished inference, even sleep(0).
    assert not delivered
    plan.end(0)
    plan.rotate()  # Index reset must not cancel a queued sentence's face.
    release.set()
    await drain()
    await drain()
    assert len(delivered) == 1
    assert delivered[0]["streamId"] == plan.stream_id
    assert delivered[0]["startOffset"] == 0
    assert delivered[0]["endOffset"] == 3
    assert delivered[0]["waitMs"] == 0
    await plan.close()


async def test_late_faces_keep_offsets_across_merged_sentences_and_rotation():
    release = [asyncio.Event(), asyncio.Event()]
    delivered = []

    async def predict(text, previous):
        await release[bool(previous)].wait()
        return {"ParamAngleX": len(previous)}

    async def send(cue):
        delivered.append(cue)

    plan = SpeechFacialPlan(PROFILE, predict)
    plan.set_sender(send)
    plan.append("你好，")
    plan.append("世界。")
    await drain()  # Start the merged lookahead before the provider identifies it.
    plan.begin(0, "你好，世界。")
    assert await plan.before_audio(0, "") is None
    plan.end(0)
    plan.rotate()
    plan.append("你好，世界。")
    assert await plan.before_audio(0, "你好，世界。") is None
    release[1].set()
    await drain()
    await drain()
    assert [(x["startOffset"], x["endOffset"]) for x in delivered] == [(6, 12)]
    release[0].set()
    await drain()
    await drain()
    assert [(x["startOffset"], x["endOffset"]) for x in delivered] == [(6, 12), (0, 6)]
    assert len({x["streamId"] for x in delivered}) == 1
    await plan.close()


async def test_audio_priority_timeout_emits_nonblocking_failure_and_close_cancels():
    delivered = asyncio.Queue()

    async def predict(text, previous):
        await asyncio.Event().wait()

    plan = SpeechFacialPlan(PROFILE, predict, timeout=0.01)
    plan.set_sender(delivered.put)
    plan.append("你好。")
    assert await plan.before_audio(0, "你好。") is None
    cue = await asyncio.wait_for(delivered.get(), 0.2)
    assert cue["error"] == "TimeoutError" and cue["waitMs"] == 0
    await plan.close()

    plan = SpeechFacialPlan(PROFILE, predict)
    plan.set_sender(delivered.put)
    plan.append("再见。")
    assert await plan.before_audio(0, "再见。") is None
    await drain()
    await plan.close()
    assert delivered.empty()
    assert not plan._deliveries


async def test_cancelling_strict_gate_never_swallows_parent_cancellation():
    async def predict(text, previous):
        await asyncio.Event().wait()

    plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    plan.append("你好。")
    gate = asyncio.create_task(plan.before_audio(0, "你好。"))
    await drain()
    gate.cancel()
    plan.cancel_pending()
    with pytest.raises(asyncio.CancelledError):
        await gate
    await plan.close()


async def test_late_face_cannot_split_audio_metadata_from_its_pcm():
    from fish_memory.tts.routes import _SerializedTtsSocket

    audio_json_sent = asyncio.Event()
    release_audio = asyncio.Event()
    sent = []

    async def send_json(item):
        sent.append(item["type"])
        if item["type"] == "audio":
            audio_json_sent.set()
            await release_audio.wait()

    async def send_bytes(item):
        sent.append("pcm")

    socket = _SerializedTtsSocket(
        SimpleNamespace(send_json=send_json, send_bytes=send_bytes)
    )
    audio = asyncio.create_task(socket.send_packet({"type": "audio"}, b"\x00\x00"))
    await audio_json_sent.wait()
    face = asyncio.create_task(socket.send_json({"type": "facial"}))
    await drain()
    assert sent == ["audio"] and not face.done()
    release_audio.set()
    await asyncio.gather(audio, face)
    assert sent == ["audio", "pcm", "facial"]


async def test_cloud_completion_does_not_retire_late_face():
    from fish_memory.tts.models import TtsStartRequest
    from fish_memory.tts.routes import _run_tts_task

    release = asyncio.Event()
    sent = []

    async def predict(text, previous):
        await release.wait()
        return {"ParamAngleX": 10}

    plan = SpeechFacialPlan(PROFILE, predict)

    def result(kind, text=None):
        output = {"type": kind, "sentence": {"index": 0}}
        if text is not None:
            output["original_text"] = text
        return json.dumps(
            {"header": {"event": "result-generated"}, "payload": {"output": output}}
        )

    class Provider:
        task_id = "task-1"
        incoming = iter(
            [
                json.dumps({"header": {"event": "task-started"}}),
                result("sentence-begin", "你好。"),
                result("sentence-synthesis"),
                b"\x00\x00",
                result("sentence-end"),
                json.dumps({"header": {"event": "task-finished"}}),
            ]
        )

        def begin_task(self):
            return self.task_id

        async def send(self, value):
            pass

        async def receive(self):
            return next(self.incoming)

    async def receive_json():
        await asyncio.Event().wait()

    async def send(value):
        sent.append(value)

    try:
        assert await asyncio.wait_for(
            _run_tts_task(
                SimpleNamespace(
                    receive_json=receive_json, send_json=send, send_bytes=send
                ),
                Provider(),
                TtsStartRequest(
                    model="qwen-audio-3.0-tts-flash", voice="longanxiaoxin", facial=True
                ),
                started_at=0.0,
                connection_reused=True,
                connection_ms=0,
                initial_text="你好。",
                facial=plan,
            ),
            0.5,
        )
        kinds = [item.get("type") if isinstance(item, dict) else "pcm" for item in sent]
        assert "finished" in kinds and "facial" not in kinds
        assert not plan._closed
        assert (
            next(
                item
                for item in sent
                if isinstance(item, dict) and item["type"] == "audio"
            )["facialWaitMs"]
            == 0
        )
        release.set()
        await drain()
        await drain()
        assert sent[-1]["type"] == "facial" and sent[-1]["endOffset"] == 3
    finally:
        await plan.close()


async def test_speech_expressions_follow_their_own_switch():
    async def evaluate(*args, **kwargs):
        raise AssertionError("switched off")

    def connection(enabled: bool):
        configuration = ExperimentalConfiguration(
            api_key="fixture", speech_facial_enabled=enabled
        )
        state = SimpleNamespace(
            speech_facial_profile=PROFILE,
            experimental_companion=SimpleNamespace(
                configuration=configuration,
                judge=SimpleNamespace(jev=SimpleNamespace(evaluate=evaluate)),
            ),
        )
        return SimpleNamespace(app=SimpleNamespace(state=state))

    assert create_speech_facial_plan(connection(False), True) is None
    plan = create_speech_facial_plan(connection(True), True)
    assert plan is not None
    await plan.close()
