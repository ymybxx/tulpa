"""Utterance-private JEV preparation, gated only at actual TTS sentence audio.

No timer or look-ahead buffer is added to the audio fast path. Speculative TTS
uses this same plan; its cues remain private until the native PCM is promoted.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fish_memory.facial_test.core import active_channels, decode_jev, questions


class FacialProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str = Field(min_length=1, max_length=100)
    modelId: str = Field(min_length=1, max_length=200)
    modelName: str = Field(min_length=1, max_length=200)
    parameters: list[dict[str, Any]] = Field(max_length=49)

    @model_validator(mode="after")
    def valid_parameters(self) -> FacialProfile:
        if self.parameters:
            active_channels(self.parameters)
        return self


@dataclass
class _Prediction:
    start: int
    text: str
    task: asyncio.Task[dict[str, float]]


class SpeechFacialPlan:
    """One reusable client utterance, including provider task rotations."""

    def __init__(
        self,
        profile: FacialProfile,
        predict: Callable,
        *,
        is_current: Callable[[], bool] = lambda: True,
        timeout: float = 2.0,
        wait_for_ready: bool = False,
    ) -> None:
        self.profile = profile
        self.stream_id = str(uuid4())
        self.wait_for_ready = wait_for_ready
        self._predict = predict
        self._is_current = is_current
        self._timeout = timeout
        self._slots = asyncio.Semaphore(2)
        self._urgent_slots = asyncio.Semaphore(1)
        self._text = ""
        self._cursor = 0
        self._predictions: dict[tuple[int, str], _Prediction] = {}
        self._running: set[tuple[int, str]] = set()
        self._sentences: dict[int | None, _Prediction] = {}
        self._sent: set[int | None] = set()
        self._deliveries: set[asyncio.Task[None]] = set()
        self._sender: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._segments: list[_Prediction] = []
        self._sent_segments: set[int] = set()
        self._closed = False

    def set_sender(self, sender: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._sender = sender

    def append(self, text: str) -> None:
        start = len(self._text)
        self._text += text
        # Most native appends are already semantic chunks. Prepare them now,
        # concurrently with continue-task, without changing TTS segmentation.
        # A provider sentence can span several appends. As soon as its ending
        # arrives, prepare the accumulated sentence too, rather than waiting
        # for sentence-begin/PCM to discover the merged text.
        pending = self._text[self._cursor :]
        for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;\n]+", pending):
            offset = self._cursor + match.start()
            if offset + len(match.group()) <= start:
                continue
            if (offset, match.group()) != (start, text):
                self._prepare(offset, match.group())
        self._prepare(start, text)

    def append_segment(self, text: str, speech_text: str) -> int:
        """Prepare exactly one face per submitted tagged TTS segment.

        Predictions see the expression tags; playback offsets and history stay
        in clean text. No extra predictions are created for provider sub-sentences.
        """
        start = len(self._text)
        self._text += text
        prediction = self._prepare(start, text, expression_text=speech_text)
        self._segments.append(prediction)
        return len(self._segments) - 1

    async def before_segment_audio(
        self, segment: int, sentence_index: int
    ) -> dict[str, Any] | None:
        if segment in self._sent_segments or self._closed or not self._is_current():
            return None
        self._sent_segments.add(segment)
        prediction = self._segments[segment]
        return await self._before_prediction_audio(
            sentence_index, prediction, segment_index=segment
        )

    def _prepare(
        self,
        start: int,
        text: str,
        *,
        urgent: bool = False,
        expression_text: str | None = None,
    ) -> _Prediction:
        key = (start, expression_text if expression_text is not None else text)
        cached = self._predictions.get(key)
        if urgent and cached and not cached.task.done() and key not in self._running:
            # The sentence about to play takes precedence over queued lookahead.
            cached.task.cancel()
            del self._predictions[key]
        if key not in self._predictions:
            previous = self._text[max(0, start - 1200) : start]

            async def request() -> dict[str, float]:
                self._running.add(key)
                async with asyncio.timeout(self._timeout):
                    return await self._predict(
                        expression_text if expression_text is not None else text,
                        previous,
                    )

            async def run() -> dict[str, float]:
                # Two speculative requests, plus at most one current sentence.
                # Future sentences do not expire merely while queued; each
                # actual request and the audio gate have bounded deadlines.
                if urgent:
                    async with self._urgent_slots:
                        return await request()
                async with self._slots:
                    return await request()

            task = asyncio.create_task(run(), name="speech-facial-jev")
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            self._predictions[key] = _Prediction(start, text, task)
        return self._predictions[key]

    def begin(self, index: int | None, text: str) -> None:
        if not text:
            return
        existing = self._sentences.get(index)
        if existing is not None and existing.text == text:
            return
        start = self._text.find(text, self._cursor)
        if start < 0:
            return  # Unknown provider text must not be attached to another line.
        self._cursor = start + len(text)
        self._sentences[index] = self._prepare(start, text, urgent=True)
        self._sent.discard(index)

    def end(self, index: int | None) -> None:
        self._sentences.pop(index, None)
        self._sent.discard(index)

    async def before_audio(self, index: int | None, text: str) -> dict[str, Any] | None:
        if index in self._sent or self._closed or not self._is_current():
            return None
        if index not in self._sentences:
            self.begin(index, text)
        prediction = self._sentences.get(index)
        if prediction is None:
            return None
        self._sent.add(index)
        return await self._before_prediction_audio(index, prediction)

    async def _before_prediction_audio(
        self,
        index: int | None,
        prediction: _Prediction,
        *,
        segment_index: int | None = None,
    ) -> dict[str, Any] | None:
        if not self.wait_for_ready and not prediction.task.done():
            # Audio-priority mode never yields to pending inference. Delivery
            # belongs to the socket/utterance, not the provider synthesis task:
            # it may arrive after `finished` while device PCM is still playing.
            task = asyncio.create_task(
                self._deliver_later(index, prediction, segment_index=segment_index),
                name="speech-facial-delivery",
            )
            self._deliveries.add(task)
            task.add_done_callback(self._deliveries.discard)
            return None
        return await self._resolve(
            index,
            prediction,
            measure_wait=self.wait_for_ready,
            segment_index=segment_index,
        )

    async def _deliver_later(
        self,
        index: int | None,
        prediction: _Prediction,
        *,
        segment_index: int | None = None,
    ) -> None:
        try:
            cue = await self._resolve(
                index, prediction, measure_wait=False, segment_index=segment_index
            )
            if cue is not None and self._sender is not None:
                await self._sender(cue)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Socket shutdown must not create an unhandled task exception.
            # The socket owner closes all remaining deliveries in its finally.
            self.cancel_pending()

    async def _resolve(
        self,
        index: int | None,
        prediction: _Prediction,
        *,
        measure_wait: bool,
        segment_index: int | None = None,
    ) -> dict[str, Any] | None:
        wait_ms = 0.0
        targets = None
        error = None
        try:
            if prediction.task.done():
                # Important: no await, sleep, fixed cushion or animation wait.
                targets = prediction.task.result()
            else:
                started = time.perf_counter()
                try:
                    targets = await asyncio.wait_for(
                        asyncio.shield(prediction.task), self._timeout
                    )
                finally:
                    if measure_wait:
                        wait_ms = (time.perf_counter() - started) * 1000
        except (TimeoutError, ValueError, RuntimeError) as exc:
            error = type(exc).__name__
            if not prediction.task.done():
                prediction.task.cancel()
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                raise
            if prediction.task.cancelled():
                return None
            raise
        except Exception as exc:  # Provider failure falls back to existing speech.
            error = type(exc).__name__
        if self._closed or not self._is_current():
            return None
        return {
            "type": "facial",
            "streamId": self.stream_id,
            "profileId": self.profile.revision,
            "sentenceIndex": index,
            **({"segmentIndex": segment_index} if segment_index is not None else {}),
            "startOffset": prediction.start,
            "endOffset": prediction.start + len(prediction.text),
            "text": prediction.text,
            "targets": targets,
            "waitMs": round(wait_ms, 3),
            "error": error,
        }

    def rotate(self) -> None:
        self._sentences.clear()
        self._sent.clear()
        # In-flight deliveries retain their captured range across provider
        # index resets. A repeated line has a different cumulative offset.
        for key in list(self._predictions):
            if key[0] < self._cursor and self._predictions[key].task.done():
                self._predictions.pop(key)
                self._running.discard(key)

    def cancel_pending(self) -> None:
        self._closed = True
        for prediction in self._predictions.values():
            if not prediction.task.done():
                prediction.task.cancel()
        for delivery in self._deliveries:
            if delivery is not asyncio.current_task():
                delivery.cancel()

    async def close(self) -> None:
        tasks: list[asyncio.Task[Any]] = [
            prediction.task for prediction in self._predictions.values()
        ]
        tasks.extend(self._deliveries)
        self.cancel_pending()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._predictions.clear()


# Each spoken sentence varies the steps the decision model was unsure of.
_VARIATION = random.Random()


def create_speech_facial_plan(
    connection: Any, enabled: bool, *, wait_for_ready: bool | None = None
) -> SpeechFacialPlan | None:
    if not enabled:
        return None
    state = connection.app.state
    profile = getattr(state, "speech_facial_profile", None)
    experiment = getattr(state, "experimental_companion", None)
    if not enabled or not profile or not profile.parameters or experiment is None:
        return None
    configuration = experiment.configuration.model_copy()
    if (
        not configuration.speech_facial_enabled
        or configuration.api_key is None
        or not configuration.api_key.get_secret_value().strip()
    ):
        return None
    channels = active_channels(profile.parameters)
    rubric = questions(channels)

    async def predict(text: str, previous: str) -> dict[str, float]:
        # Facial scores are a Jev-only question type.
        raw = await experiment.judge.jev.evaluate(
            configuration,
            {
                "previousSegments": [previous] if previous else [],
                "currentSegment": text,
            },
            rubric,
            kind="speech.facial.jev",
        )
        return decode_jev(raw, channels, _VARIATION)

    return SpeechFacialPlan(
        profile,
        predict,
        is_current=lambda: getattr(state, "speech_facial_profile", None) is profile,
        wait_for_ready=(
            bool(
                getattr(
                    getattr(state, "settings", None),
                    "speech_facial_wait_for_ready",
                    False,
                )
            )
            if wait_for_ready is None
            else wait_for_ready
        ),
    )
