"""Tagged transport uses fake provider PCM only; never calls TTS or JEV APIs."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from fish_memory.tts.facial import FacialProfile, SpeechFacialPlan
from fish_memory.tts.models import TtsStartRequest
from fish_memory.tts.routes import _run_tts_task, _validate_speech_tag_request
from fish_memory.tts.tagged_stream import _Segment, _SentenceRanges

PROFILE = FacialProfile(
    revision="v1",
    modelId="cat",
    modelName="猫",
    parameters=[
        {"id": "ParamAngleX", "minimum": -30, "maximum": 30, "defaultValue": 0}
    ],
)


def event(kind, index=0, text=None):
    if kind in {"task-started", "task-finished"}:
        return json.dumps({"header": {"event": kind}})
    output = {"type": kind, "sentence": {"index": index}}
    if text is not None:
        output["original_text"] = text
    return json.dumps(
        {"header": {"event": "result-generated"}, "payload": {"output": output}}
    )


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []

    async def receive_json(self):
        return await self.incoming.get()

    async def send_json(self, value):
        self.sent.append(value)

    async def send_bytes(self, value):
        self.sent.append(value)

    def submit(self, text, display_text=None):
        self.incoming.put_nowait(
            {"type": "append", "text": text, "display_text": display_text}
        )

    async def until(self, predicate):
        async with asyncio.timeout(0.5):
            while not predicate():  # noqa: ASYNC110 - deterministic fake task drain
                await asyncio.sleep(0)

    def events(self, kind):
        return [x for x in self.sent if isinstance(x, dict) and x["type"] == kind]


class Provider:
    def __init__(self, *, gate=None):
        self.task_id = ""
        self.count = 0
        self.messages = []
        self.incoming = asyncio.Queue()
        self.tasks = {}
        self.gate = gate
        self.emissions = []

    def begin_task(self):
        self.count += 1
        self.task_id = str(self.count)
        self.tasks[self.task_id] = []
        return self.task_id

    async def send(self, message):
        self.messages.append(message)
        task_id = message["header"]["task_id"]
        action = message["header"]["action"]
        if action == "run-task":
            self.incoming.put_nowait(event("task-started"))
        elif action == "continue-task":
            self.tasks[task_id].append(message["payload"]["input"]["text"])
        elif message["payload"]["input"].get("directive") == "cancel":
            for task in self.emissions:
                task.cancel()
        else:
            self.emissions.append(asyncio.create_task(self.emit(task_id)))

    async def emit(self, task_id):
        if self.gate is not None and self.tasks[task_id]:
            await self.gate.wait()
        for text in self.tasks[task_id]:
            # The real provider can emit a separate zero-text laughter sentence.
            parts = [text]
            if text.endswith("[laughing]") and text != "[laughing]":
                parts = [text[: -len("[laughing]")], "[laughing]"]
            for index, part in enumerate(parts):
                for item in [
                    event("sentence-begin", index, part),
                    event("sentence-synthesis", index),
                    b"\x01\x00",
                    event("sentence-end", index),
                ]:
                    self.incoming.put_nowait(item)
        self.incoming.put_nowait(event("task-finished"))

    async def receive(self):
        return await self.incoming.get()


def run(socket, provider, **kwargs):
    return _run_tts_task(
        socket,
        provider,
        TtsStartRequest(
            model="qwen-audio-3.0-tts-flash",
            voice="longanxiaoxin",
            speech_tags=True,
            facial=True,
        ),
        started_at=time.perf_counter(),
        connection_reused=True,
        connection_ms=None,
        **kwargs,
    )


async def test_tagged_segments_preserve_clean_unicode_ranges_and_one_face_each():
    calls = []

    async def predict(text, previous):
        calls.append((text, previous))
        return {"ParamAngleX": 1}

    face = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    socket, provider = Socket(), Provider()
    socket.submit("[excited]你😀好。[laughing]", "你😀好。")
    socket.submit("[whispers]你😀好。", "你😀好。")
    socket.incoming.put_nowait({"type": "finish"})
    try:
        assert await asyncio.wait_for(run(socket, provider, facial=face), 0.5)
        audio = socket.events("audio")
        assert [(x["text"], x["startOffset"], x["endOffset"]) for x in audio] == [
            ("你😀好。", 0, 4),
            ("", 4, 4),
            ("你😀好。", 4, 8),
        ]
        assert [x["sentenceIndex"] for x in audio] == [0, 1, 2]
        assert [x["segmentIndex"] for x in audio] == [0, 0, 1]
        assert calls == [
            ("[excited]你😀好。[laughing]", ""),
            ("[whispers]你😀好。", "你😀好。"),
        ]
        faces = socket.events("facial")
        assert [(x["text"], x["startOffset"], x["endOffset"]) for x in faces] == [
            ("你😀好。", 0, 4),
            ("你😀好。", 4, 8),
        ]
        assert all(x["waitMs"] == 0 for x in faces)
        assert len(socket.events("finished")) == 1
        assert len(socket.events("segment_finished")) == 2
        assert all(len(texts) == 1 for texts in provider.tasks.values())
    finally:
        await face.close()


async def test_clause_tags_share_one_provider_task_and_one_face_per_source_segment():
    calls = []

    async def predict(text, previous):
        calls.append((text, previous))
        return {"ParamAngleX": 1}

    face = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    socket, provider = Socket(), Provider()
    first = "[whispers]伊利丹，我现在饿得直抽抽。"
    second = "连口热饭都没有，好惨啊呜呜[laughing]"
    socket.submit(first)
    socket.submit(second)
    socket.incoming.put_nowait({"type": "finish"})
    try:
        assert await asyncio.wait_for(run(socket, provider, facial=face), 0.5)
        assert list(provider.tasks.values()) == [
            ["[whispers]伊利丹，[whispers]我现在饿得直抽抽。"],
            ["[whispers]连口热饭都没有，[whispers]好惨啊呜呜[laughing]"],
        ]
        assert calls == [
            (first, ""),
            ("[whispers]" + second, "伊利丹，我现在饿得直抽抽。"),
        ]
        assert len(socket.events("facial")) == 2
        assert [
            (x["text"], x["startOffset"], x["endOffset"])
            for x in socket.events("audio")
        ] == [
            ("伊利丹，我现在饿得直抽抽。", 0, 13),
            ("连口热饭都没有，好惨啊呜呜", 13, 26),
            ("", 26, 26),
        ]
    finally:
        await face.close()


async def test_empty_task_prestarts_and_future_jev_does_not_wait_for_current_audio():
    gate = asyncio.Event()
    calls = []

    async def predict(text, previous):
        calls.append(text)
        return {"ParamAngleX": 0}

    socket, provider = Socket(), Provider(gate=gate)
    face = SpeechFacialPlan(PROFILE, predict)
    task = asyncio.create_task(run(socket, provider, facial=face))
    try:
        await socket.until(lambda: bool(socket.events("ready")))
        assert provider.count == 1 and provider.tasks["1"] == []
        socket.submit("[excited]第一句。", "第一句。")
        await socket.until(lambda: bool(provider.emissions))
        socket.submit("[serious]第二句。", "第二句。")
        await socket.until(lambda: len(calls) == 2)
        assert not socket.events("audio") and provider.count == 1
        gate.set()
        await socket.until(lambda: len(socket.events("segment_finished")) == 2)
        # The third task is started while waiting for future text, not afterwards.
        await socket.until(lambda: provider.count == 3)
        assert provider.tasks["3"] == []
        socket.incoming.put_nowait({"type": "finish"})
        assert await asyncio.wait_for(task, 0.5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await face.close()


@pytest.mark.parametrize("wait_for_ready", [False, True])
async def test_ready_policy_still_gates_only_corresponding_segment(wait_for_ready):
    release = asyncio.Event()

    async def predict(text, previous):
        await release.wait()
        return {"ParamAngleX": 2}

    face = SpeechFacialPlan(PROFILE, predict, wait_for_ready=wait_for_ready)
    socket, provider = Socket(), Provider()
    socket.submit("[excited]你好。", "你好。")
    socket.incoming.put_nowait({"type": "finish"})
    task = asyncio.create_task(run(socket, provider, facial=face))
    try:
        await socket.until(lambda: bool(socket.events("sentence_begin")))
        if wait_for_ready:
            for _ in range(10):
                await asyncio.sleep(0)
            assert not socket.events("audio")
        else:
            await socket.until(lambda: bool(socket.events("audio")))
            assert socket.events("audio")[0]["facialWaitMs"] == 0
            assert not socket.events("facial")
        release.set()
        assert await asyncio.wait_for(task, 0.5)
        await socket.until(lambda: bool(socket.events("facial")))
        assert socket.events("facial")[0]["text"] == "你好。"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await face.close()


async def test_stop_interrupts_provider_and_pending_face_even_after_finish():
    gate = asyncio.Event()

    async def predict(text, previous):
        await gate.wait()
        return {}

    socket, provider = Socket(), Provider(gate=gate)
    face = SpeechFacialPlan(PROFILE, predict, wait_for_ready=True)
    socket.submit("[excited]你好。", "你好。")
    socket.incoming.put_nowait({"type": "finish"})
    task = asyncio.create_task(run(socket, provider, facial=face))
    try:
        await socket.until(lambda: bool(provider.emissions))
        socket.incoming.put_nowait({"type": "stop"})
        assert await asyncio.wait_for(task, 0.5) is False
        assert socket.events("finished")[0]["stopped"] is True
        assert not socket.events("audio") and not socket.events("facial")
        assert face._closed
        assert provider.messages[-1]["payload"]["input"] == {"directive": "cancel"}
    finally:
        await face.close()


async def test_pure_sound_does_not_advance_history():
    socket, provider = Socket(), Provider()
    socket.submit("[laughing]", "")
    socket.incoming.put_nowait({"type": "finish"})
    assert await asyncio.wait_for(run(socket, provider), 0.5)
    audio = socket.events("audio")[0]
    assert audio["text"] == "" and audio["startOffset"] == audio["endOffset"] == 0


async def test_display_text_mismatch_rejected_before_submission():
    socket, provider = Socket(), Provider()
    socket.submit("[excited]你好。", "不同正文。")
    with pytest.raises(ValueError, match="display_text must match"):
        await asyncio.wait_for(run(socket, provider), 0.5)
    assert not any(provider.tasks.values())


async def test_idle_empty_task_retires_without_repeated_empty_requests(monkeypatch):
    monkeypatch.setattr(
        "fish_memory.tts.routes.TTS_IDLE_ROTATION_INTERVAL_SECONDS", 0.01
    )
    socket, provider = Socket(), Provider()
    task = asyncio.create_task(run(socket, provider))
    try:
        await socket.until(lambda: bool(provider.emissions))
        await asyncio.sleep(0.025)
        assert provider.count == 1
        socket.submit("[excited]终于来了。", "终于来了。")
        socket.incoming.put_nowait({"type": "finish"})
        assert await asyncio.wait_for(task, 0.5)
        assert provider.count == 2
        assert socket.events("audio")[0]["text"] == "终于来了。"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "model,enabled,tagged",
    [
        ("qwen-audio-3.0-tts-flash", False, False),
        ("qwen-audio-3.0-tts-flash", True, True),
        ("cosyvoice-v3-flash", False, False),
    ],
)
async def test_tagged_route_requires_explicit_flag_and_capability(
    monkeypatch,
    model,
    enabled,
    tagged,
):
    calls = []

    async def run_tagged(*args, **kwargs):
        calls.append("tagged")
        return True

    async def run_original(*args, **kwargs):
        calls.append("original")
        return True

    monkeypatch.setattr("fish_memory.tts.routes.run_tagged_tts", run_tagged)
    monkeypatch.setattr("fish_memory.tts.routes._run_tts_task_prepared", run_original)
    assert await _run_tts_task(
        SimpleNamespace(),
        SimpleNamespace(),
        TtsStartRequest(model=model, voice="voice", speech_tags=enabled),
        started_at=0,
        connection_reused=True,
        connection_ms=None,
    )
    assert calls == ["tagged" if tagged else "original"]


@pytest.mark.parametrize(
    "provider,model",
    [("local", "qwen-audio-3.0-tts-flash"), ("dashscope", "cosyvoice-v3-flash")],
)
def test_unsupported_tag_request_is_rejected_before_provider_start(provider, model):
    request = TtsStartRequest(model=model, voice="voice", speech_tags=True)
    with pytest.raises(ValueError, match="speech tags are not supported"):
        _validate_speech_tag_request(request, provider)
    _validate_speech_tag_request(
        request.model_copy(update={"speech_tags": False}), provider
    )


async def test_unsupported_tag_model_cannot_reach_legacy_stream(monkeypatch):
    async def run_original(*args, **kwargs):
        raise AssertionError("tagged text must not enter unsupported TTS")

    monkeypatch.setattr("fish_memory.tts.routes._run_tts_task_prepared", run_original)
    with pytest.raises(ValueError, match="speech tags are not supported"):
        await _run_tts_task(
            SimpleNamespace(),
            SimpleNamespace(),
            TtsStartRequest(
                model="cosyvoice-v3-flash", voice="voice", speech_tags=True
            ),
            started_at=0,
            connection_reused=True,
            connection_ms=None,
        )


def test_mismatched_provider_sentence_cannot_skip_unspoken_prefix():
    segment = _Segment(0, "[serious]前文。后文。", "前文。后文。", 4, None)
    ranges = _SentenceRanges(segment, [0])
    unknown = ranges.get(0, "后文。")
    assert (unknown["text"], unknown["startOffset"], unknown["endOffset"]) == ("", 4, 4)
    actual = ranges.get(1, "[serious]前文。")
    assert (actual["text"], actual["startOffset"], actual["endOffset"]) == (
        "前文。",
        4,
        7,
    )
