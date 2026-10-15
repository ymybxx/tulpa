import asyncio
import json
from dataclasses import replace
from uuid import UUID

import pytest

from fish_memory.companion.draft_tts import DraftTtsSession
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.tts import ApiTtsOutput, TtsConfiguration
from fish_memory.companion.voice_drafts import VoiceDraftKey
from fish_memory.tts.speech_tags import strip_speech_tags
from tests.companion.test_draft_tts import (
    KEY,
    FakeDraftClient,
    FakeDraftOutput,
    draft_snapshot,
)
from tests.companion.test_proactive_streaming import StreamingApi
from tests.companion.test_session import (
    ANSWER_ID,
    FISH_ID,
    FakeApi,
    FakeTts,
    proactive_message,
    wait_until,
)
from tests.companion.test_tts import FakeSink, FakeSocket


class TaggedTts(FakeTts):
    def __init__(self):
        super().__init__()
        self.tags = False
        self.modes = []

    def set_speech_tags(self, enabled):
        self.tags = enabled
        self.modes.append(enabled)

    async def append(self, text):
        self.calls.append(("append", text))
        self.played_offset += len(strip_speech_tags(text) if self.tags else text)


@pytest.mark.asyncio
async def test_chat_tags_split_across_deltas_and_next_turn_plain():
    api, tts = FakeApi(), TaggedTts()
    session = CompanionSession(api, tts)
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        for tagged in (True, False):
            await session.submit_transcript("说句话")
            chunks = [
                ("", "[exc"),
                ("你好。", "[excited]你好。"),
                ("", "[excited]你好。[laughing]"),
            ]
            if not tagged:
                chunks = [("你好。", "")]
            for clean, raw in chunks:
                await api.answer_events.put(
                    {
                        "event": "answer.delta",
                        "data": {
                            "messageId": str(ANSWER_ID),
                            "content": clean,
                            **(
                                {"speechTags": True, "speechContent": raw}
                                if tagged
                                else {}
                            ),
                        },
                    }
                )
            await api.answer_events.put(
                {
                    "event": "answer.completed",
                    "data": {"messageId": str(ANSWER_ID)},
                }
            )
            await asyncio.wait_for(session.wait_for_idle(), 1)
        assert [text for op, text in tts.calls if op == "append"] == [
            "[excited]你好。[laughing]",
            "你好。",
        ]
        assert tts.modes == [True, False]
        assert session._assistant_text == "你好。"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_proactive_initial_and_stream_use_same_tagged_transcript():
    class Api(StreamingApi):
        async def iter_proactive_expression(self, *args, **kwargs):
            yield {
                "text": "继续检查。",
                "speechContent": "[excited]修好了。[serious]继续检查。",
            }

    api, tts = Api(), TaggedTts()
    session = CompanionSession(api, tts)
    message = replace(
        proactive_message(2, content="修好了。"),
        streaming=True,
        speech_tags=True,
        speech_content="[excited]修好了。",
    )
    try:
        await session.set_binding(FISH_ID)
        await api.agent_events.put(message)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert [text for op, text in tts.calls if op == "append"] == [
            "[excited]修好了。",
            "[serious]继续检查。",
        ]
        assert api.agent_consumption_calls == [
            (message.message_id, len("修好了。继续检查。"), False)
        ]
    finally:
        await session.close()


class TaggedSocket(FakeSocket):
    async def send(self, value):
        message = json.loads(value)
        self.sent.append(message)
        action = message["type"]
        if action == "start":
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "ready",
                        "provider": "dashscope",
                        "model": message["model"],
                        "voice": message["voice"],
                        "sampleRate": message["sample_rate"],
                        "facialStreamId": "speech-1",
                    }
                )
            )
        elif action == "append":
            text = message.get("display_text", message["text"])
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "facial",
                        "profileId": "pet",
                        "streamId": "speech-1",
                        "segmentIndex": 0,
                        "text": text,
                        "startOffset": 0,
                        "endOffset": len(text),
                        "targets": {"ParamEyeLSmile": 1},
                    }
                )
            )
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "audio",
                        "text": text,
                        "sentenceIndex": 0,
                        "segmentIndex": 0,
                        "startOffset": 0,
                        "endOffset": len(text),
                    }
                )
            )
            await self.incoming.put(b"\x01\x00" * 100)
            await self.incoming.put(
                json.dumps({"type": "segment_finished", "segmentIndex": 0})
            )
        elif action in {"finish", "stop"}:
            await self.incoming.put(json.dumps({"type": "finished"}))


@pytest.mark.asyncio
async def test_sound_only_face_waits_for_device_bytes_and_drops_after_sound():
    class Sink(FakeSink):
        played_audio_bytes = 0

    socket, sink = TaggedSocket(), Sink()

    async def connect(_):
        return socket

    output = ApiTtsOutput(
        "http://local", sink=sink, facial=True, websocket_connector=connect
    )
    output.set_speech_tags(True)
    try:
        await output.start(text_offset=9)
        await output.append("[laughing]")
        await wait_until(lambda: bool(output._finished_audio_segments))
        assert socket.sent[1]["display_text"] == ""
        assert sink.calls[-1] == ("append", (b"\x01\x00" * 100, 9, 9))
        assert output.pop_facial_cue() is None
        sink.played_audio_bytes = 2
        assert output.pop_facial_cue() == {
            "profileId": "pet",
            "targets": {"ParamEyeLSmile": 1},
        }
        # A prediction arriving only after the sound ended cannot animate later speech.
        output._facial_cues.append((9, 9, {"_segment": ("speech-1", 0), "targets": {}}))
        sink.played_audio_bytes = 200
        assert output.pop_facial_cue() is None
        assert output.played_offset == 9
    finally:
        await output.close_transport()


@pytest.mark.asyncio
async def test_voice_change_before_start_downgrades_queued_markup():
    socket, sink = FakeSocket(), FakeSink()

    async def connect(_):
        return socket

    async def refresh():
        output.configure(model="matcha-icefall-zh-en", voice="default")
        output.set_speech_tag_capability(False)

    output = ApiTtsOutput(
        "http://local", sink=sink, websocket_connector=connect, before_start=refresh
    )
    output.set_speech_tags(True)
    try:
        await output.start()
        await output.append("[excited]你好。[laughing]")
        await output.finish()
        assert "speech_tags" not in socket.sent[0]
        assert socket.sent[1] == {"type": "append", "text": "你好。"}
        assert output.played_offset == 3
    finally:
        await output.close_transport()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["matching", "different_style", "stale_cursor"])
@pytest.mark.parametrize(
    "raw",
    [
        "[excited]修好了。[laughing]",
        "[disappointed]修好了。[laughing]",
        "[laughing]",
    ],
)
async def test_draft_promotes_only_matching_marked_prefix(scenario, raw):
    config = TtsConfiguration("qwen-audio-3.0-tts-flash", "voice", 16000, True)
    client, sink = FakeDraftClient(), FakeSink()
    outputs = []

    class Output(FakeDraftOutput):
        def set_speech_tags(self, enabled):
            self.configuration = replace(self.configuration, speech_tags=enabled)
            self.provider_state.configuration = self.configuration

        async def append(self, text):
            self.calls.append(("append", text))
            await self.sink.append(
                b"\x01\x00" * 20,
                start_offset=0,
                end_offset=len(strip_speech_tags(text)),
            )

    async def load():
        return config

    def factory(private):
        output = Output(private)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=sink,
        max_buffer_bytes=1000,
        output_factory=factory,
        load_configuration=load,
    )
    clean = strip_speech_tags(raw)
    try:
        assert await draft.start(KEY, revision=4)
        snapshot = draft_snapshot(clean, flush_offset=len(clean))
        snapshot["data"].update(speechTags=True, speechContent=raw)
        await client.events.put(snapshot)
        await wait_until(lambda: draft._submission_complete)
        assert sink.calls == []
        if scenario == "stale_cursor":
            draft._submitted_speech = "[sad]上一轮。"
        promotion = await draft.promote(
            KEY,
            revision=4,
            content=clean,
            flush_offset=len(clean),
            speech_content="[sad]" + clean if scenario == "different_style" else raw,
        )
        if scenario == "matching":
            assert promotion is not None
            assert promotion.submitted_offset == len(clean)
            assert promotion.speech_offset == len(raw)
            assert sum(op == "append" for op, _ in outputs[0].calls) == 1
        else:
            assert promotion is None
            assert not any(op == "append" for op, _ in sink.calls)
    finally:
        await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_previous", [True, False])
async def test_new_tagged_turn_adopts_ready_provider_without_repeating_text(
    cancel_previous,
):
    """A prior turn's raw tag cursor must not contaminate an unsent draft."""
    config = TtsConfiguration("qwen-audio-3.0-tts-flash", "voice", 16000, True)
    client, sink = FakeDraftClient(), FakeSink()
    outputs = []

    class Output(FakeDraftOutput):
        def set_speech_tags(self, enabled):
            self.configuration = replace(self.configuration, speech_tags=enabled)
            self.provider_state.configuration = self.configuration

    def factory(private):
        output = Output(private, emit_pcm=False)
        outputs.append(output)
        return output

    async def load():
        return config

    api, ordinary_tts, arbiter = FakeApi(), TaggedTts(), PlaybackArbiter()
    draft = DraftTtsSession(
        client=client,
        live_sink=sink,
        max_buffer_bytes=1000,
        output_factory=factory,
        load_configuration=load,
        playback_arbiter=arbiter,
    )
    session = CompanionSession(
        api, ordinary_tts, draft_tts=draft, playback_arbiter=arbiter
    )
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        key = VoiceDraftKey(FISH_ID, session.session_id, KEY.turn_id)
        previous_key = replace(key, turn_id=UUID(int=99))
        assert await draft.start(previous_key, revision=4)
        previous = draft_snapshot("上一句话。", flush_offset=5)
        previous["data"].update(
            turnId=str(previous_key.turn_id),
            speechTags=True,
            speechContent="[sad]上一句话。",
        )
        await client.events.put(previous)
        await wait_until(lambda: draft._submission_complete)
        if cancel_previous:
            await draft.cancel(previous_key, through_revision=4)

        # The next provider is ready, but its private text has not arrived yet.
        assert await draft.start(key, revision=4)
        assert await session.submit_voice_turn(
            key.turn_id, revision=5, promote_revision=4, content="滚滚滚。"
        )
        clean = "好嘛我走就是了……"
        await api.answer_events.put(
            {
                "event": "answer.snapshot",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "content": clean,
                    "flushOffset": len(clean),
                    "speechTags": True,
                    "speechContent": "[trembling]" + clean,
                },
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {"messageId": str(ANSWER_ID)},
            }
        )
        await asyncio.wait_for(session.wait_for_idle(), 1)
        submitted = [
            text
            for op, text in outputs[-1].calls + ordinary_tts.calls
            if op == "append"
        ]
        assert "".join(map(strip_speech_tags, submitted)) == clean
        assert ordinary_tts.calls == []
        assert outputs[-1].calls.count(("start", None)) == 1
    finally:
        await session.close()
        await draft.close()
