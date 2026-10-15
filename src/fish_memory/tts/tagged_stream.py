"""Single-input Qwen speech tags with independent intake and face preparation."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from starlette.websockets import WebSocketDisconnect

from fish_memory.tts.facial import SpeechFacialPlan
from fish_memory.tts.models import TtsAppendRequest, TtsStartRequest
from fish_memory.tts.provider import (
    build_cancel_task,
    build_continue_task,
    build_finish_task,
    build_run_task,
    parse_server_event,
)
from fish_memory.tts.speech_tags import SpeechTagRepeater, strip_speech_tags


@dataclass(frozen=True)
class _Segment:
    index: int
    text: str
    display_text: str
    start: int
    face: int | None
    provider_text: str | None = None


class _SentenceRanges:
    """Map provider sentences to clean offsets, including zero-width sounds."""

    def __init__(self, segment: _Segment, sequence: list[int]) -> None:
        self.segment = segment
        self.sequence = sequence
        self.cursor = 0
        self.ranges: dict[int | None, dict[str, Any]] = {}
        self.active: dict[str, Any] | None = None

    def get(self, index: int | None, text: str | None) -> dict[str, Any]:
        if index in self.ranges:
            return self.ranges[index]
        if index is None and self.active is not None and not text:
            return self.active
        clean = strip_speech_tags(text or "")
        start = self.cursor
        if clean:
            if self.segment.display_text.startswith(clean, self.cursor):
                self.cursor += len(clean)
            else:
                # Never advance history for unknown or normalized provider text.
                clean = ""
        result = {
            "sentenceIndex": self.sequence[0],
            "segmentIndex": self.segment.index,
            "text": clean,
            "startOffset": self.segment.start + start,
            "endOffset": self.segment.start + start + len(clean),
        }
        self.sequence[0] += 1
        self.ranges[index] = result
        return result


async def _started(session: Any) -> None:
    while True:
        raw = await session.receive()
        if not isinstance(raw, str):
            continue
        event = parse_server_event(json.loads(raw))
        if event.kind == "failed":
            raise event.failure()
        if event.kind == "started":
            return


async def _forward_segment(
    websocket: Any,
    session: Any,
    segment: _Segment | None,
    sequence: list[int],
    facial: SpeechFacialPlan | None,
    *,
    started_at: float,
    timing: dict[str, float | None],
) -> None:
    ranges = _SentenceRanges(segment, sequence) if segment is not None else None
    pending = None
    face_checked = False
    while True:
        raw = await session.receive()
        if isinstance(raw, bytes):
            if pending is None or ranges is None or segment is None:
                continue
            now = time.perf_counter()
            first = timing["first_audio_at"] is None
            if first:
                timing["first_audio_at"] = now
            source = ranges.get(pending.sentence_index, pending.text)
            wait_ms = 0.0
            if facial is not None and segment.face is not None and not face_checked:
                face_checked = True
                cue = await facial.before_segment_audio(
                    segment.face, source["sentenceIndex"]
                )
                if cue is not None:
                    wait_ms = cue["waitMs"]
                    await websocket.send_json(cue)
            await websocket.send_packet(
                {
                    "type": "audio",
                    **source,
                    "firstPacketMs": round(
                        (now - (timing["text_sent_at"] or started_at)) * 1000, 1
                    )
                    if first
                    else None,
                    "firstPacketFromStartMs": round((now - started_at) * 1000, 1)
                    if first
                    else None,
                    "facialWaitMs": wait_ms if first else None,
                },
                raw,
            )
            pending = None
            continue
        if not isinstance(raw, str):
            continue
        event = parse_server_event(json.loads(raw))
        if event.kind == "failed":
            raise event.failure()
        if event.kind == "finished":
            return
        if event.kind == "audio":
            pending = event
        elif event.kind in {"sentence_begin", "sentence_end"} and ranges is not None:
            source = ranges.get(event.sentence_index, event.text)
            ranges.active = source if event.kind == "sentence_begin" else None
            await websocket.send_json({"type": event.kind, **source})


async def run_tagged_tts(
    websocket: Any,
    session: Any,
    request: TtsStartRequest,
    *,
    started_at: float,
    connection_reused: bool,
    connection_ms: float | None,
    initial_text: str | None,
    initial_display_text: str | None,
    initial_text_is_final: bool,
    facial: SpeechFacialPlan | None,
    idle_timeout: float,
) -> bool:
    """Each append is complete; receive future segments while synthesizing.

    An empty provider task starts before its text is ready. After 10 seconds
    without text, retire it rather than reaching DashScope's idle timeout.
    Audio never waits for future text, a fixed buffer, or a non-blocking face.
    """
    queue: asyncio.Queue[_Segment | None] = asyncio.Queue()
    stopped = asyncio.Event()
    timing: dict[str, float | None] = {
        "text_sent_at": None,
        "first_audio_at": None,
    }
    offset = 0
    count = 0
    tags = SpeechTagRepeater()
    if facial is not None:
        facial.set_sender(websocket.send_json)

    def enqueue(text: str, display_text: str | None) -> None:
        nonlocal offset, count
        clean = strip_speech_tags(text)
        if display_text is not None and display_text != clean:
            raise ValueError("display_text must match text without speech tags")
        expression_text, provider_text = tags.prepare(text)
        face = (
            facial.append_segment(clean, expression_text)
            if facial is not None
            else None
        )
        queue.put_nowait(_Segment(count, text, clean, offset, face, provider_text))
        offset += len(clean)
        count += 1

    final_received = initial_text is not None and initial_text_is_final
    if initial_text is not None:
        enqueue(initial_text, initial_display_text)
        if final_received:
            queue.put_nowait(None)

    async def receive() -> None:
        nonlocal final_received
        try:
            while True:
                message = await websocket.receive_json()
                kind = message.get("type")
                if kind == "append" and not final_received:
                    append = TtsAppendRequest.model_validate(message)
                    enqueue(append.text, append.display_text)
                elif kind == "flush":
                    continue  # Every append already commits its entire segment.
                elif kind == "finish":
                    if not final_received:
                        final_received = True
                        queue.put_nowait(None)
                elif kind == "stop":
                    stopped.set()
                    return
                else:
                    raise ValueError("message must be append, flush, finish, or stop")
        except (WebSocketDisconnect, RuntimeError):
            stopped.set()

    async def synthesize() -> None:
        first_task = True
        sequence = [0]
        pending: _Segment | None = None
        while True:
            # If finish is already queued, don't open an unnecessary empty task.
            if pending is None and not first_task and not queue.empty():
                pending = queue.get_nowait()
                if pending is None:
                    return
            task_id = session.begin_task()
            await session.send(build_run_task(request, task_id=task_id))
            await _started(session)
            if first_task:
                await websocket.send_json(
                    {
                        "type": "ready",
                        "connectMs": round(
                            (time.perf_counter() - started_at) * 1000, 1
                        ),
                        "connectionMs": round(connection_ms, 1)
                        if connection_ms is not None
                        else None,
                        "connectionReused": connection_reused,
                        "provider": "dashscope",
                        "model": request.model,
                        "voice": request.voice,
                        "sampleRate": request.sample_rate,
                        **({"facialStreamId": facial.stream_id} if facial else {}),
                    }
                )
                first_task = False
            if pending is None:
                try:
                    pending = await asyncio.wait_for(queue.get(), idle_timeout)
                except TimeoutError:
                    await session.send(build_finish_task(task_id))
                    await _forward_segment(
                        websocket,
                        session,
                        None,
                        sequence,
                        facial,
                        started_at=started_at,
                        timing=timing,
                    )
                    pending = await queue.get()
                    if pending is None:
                        return
                    # A real new segment opens a new task after the idle period.
                    continue
            if pending is not None:
                timing["text_sent_at"] = timing["text_sent_at"] or time.perf_counter()
                await session.send(
                    build_continue_task(
                        task_id,
                        pending.provider_text
                        if pending.provider_text is not None
                        else pending.text,
                    )
                )
            await session.send(build_finish_task(task_id))
            await _forward_segment(
                websocket,
                session,
                pending,
                sequence,
                facial,
                started_at=started_at,
                timing=timing,
            )
            if pending is None:
                return
            await websocket.send_json(
                {
                    "type": "segment_finished",
                    "segmentIndex": pending.index,
                    "startOffset": pending.start,
                    "endOffset": pending.start + len(pending.display_text),
                }
            )
            pending = None

    reader = asyncio.create_task(receive(), name="tagged-tts-intake")
    worker = asyncio.create_task(synthesize(), name="tagged-tts-synthesis")
    stop_wait = asyncio.create_task(stopped.wait(), name="tagged-tts-stop")
    reusable = True
    try:
        done, _ = await asyncio.wait(
            {reader, worker, stop_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if reader in done and not reader.cancelled():
            reader.result()  # Preserve validation errors instead of hiding them.
        if stopped.is_set():
            reusable = False
            if facial is not None:
                facial.cancel_pending()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            with suppress(Exception):
                await asyncio.wait_for(
                    session.send(build_cancel_task(session.task_id)), 1.0
                )
        else:
            await worker
        await websocket.send_json(
            {
                "type": "finished",
                "stopped": not reusable,
                "firstAudioMs": round(
                    (timing["first_audio_at"] - (timing["text_sent_at"] or started_at))
                    * 1000,
                    1,
                )
                if timing["first_audio_at"] is not None
                else None,
                "totalMs": round((time.perf_counter() - started_at) * 1000, 1),
            }
        )
        return reusable
    finally:
        for task in (reader, worker, stop_wait):
            if not task.done():
                task.cancel()
        await asyncio.gather(reader, worker, stop_wait, return_exceptions=True)
