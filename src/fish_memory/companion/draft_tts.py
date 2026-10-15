"""Private Companion draft TTS buffering and exact public promotion."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.diagnostics import (
    AudioDiagnosticsObserver,
    record_voice_draft_event,
)
from fish_memory.companion.draft_pcm import PromotablePcmSink as PromotablePcmSink
from fish_memory.companion.playback import (
    BoundTtsOutput,
    PlaybackArbiter,
    PlaybackOwner,
)
from fish_memory.companion.text_buffer import TtsTextBuffer
from fish_memory.companion.tts import (
    DraftTtsPending,
    PcmSink,
    TtsConfiguration,
    TtsProviderState,
)
from fish_memory.tts.speech_tags import SpeechTextBuffer, strip_speech_tags

logger = logging.getLogger(__name__)


class VoiceDraftEventClient(Protocol):
    def iter_voice_draft_events(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
    ) -> AsyncIterator[dict[str, object]]: ...


class DraftTtsOutput(Protocol):
    @property
    def configuration(self) -> TtsConfiguration: ...

    @property
    def played_offset(self) -> int: ...

    @property
    def provider_state(self) -> TtsProviderState: ...

    def configure(self, *, model: str, voice: str) -> None: ...

    async def start(self, *, text_offset: int = 0) -> None: ...

    async def append(self, text: str) -> None: ...

    async def flush(self) -> None: ...

    async def finish(self) -> None: ...

    async def stop(self) -> None: ...

    async def interrupt(self) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def duck(self) -> None: ...

    async def unduck(self) -> None: ...


DraftTtsOutputFactory = Callable[[PromotablePcmSink], DraftTtsOutput]
VoiceConfigurationLoader = Callable[[], Awaitable[TtsConfiguration]]


@dataclass(frozen=True, slots=True)
class DraftTtsPromotion:
    output: DraftTtsOutput | BoundTtsOutput
    submitted_offset: int
    speech_offset: int | None = None


@dataclass(frozen=True, slots=True)
class _DetachedDraftTts:
    output: DraftTtsOutput | None
    sink: PromotablePcmSink | None
    event_task: asyncio.Task[None] | None
    readiness_task: asyncio.Task[None] | None
    submission_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class _StartingDraftTts:
    key: VoiceDraftKey
    revision: int
    token: int
    ready: asyncio.Future[bool]


@dataclass(frozen=True, slots=True)
class _PendingDraftTts:
    token: int
    key: VoiceDraftKey
    revision: int
    output: DraftTtsOutput
    sink: PromotablePcmSink
    readiness_task: asyncio.Task[None]


class DraftTtsSession:
    """Own one revision-scoped private provider task and its PCM promotion."""

    def __init__(
        self,
        *,
        client: VoiceDraftEventClient,
        live_sink: PcmSink,
        max_buffer_bytes: int,
        output_factory: DraftTtsOutputFactory,
        load_configuration: VoiceConfigurationLoader,
        diagnostics: AudioDiagnosticsObserver | None = None,
        playback_arbiter: PlaybackArbiter | None = None,
    ) -> None:
        self._client = client
        self._live_sink = live_sink
        self._max_buffer_bytes = max_buffer_bytes
        self._output_factory = output_factory
        self._load_configuration = load_configuration
        self._diagnostics = diagnostics
        self._playback_arbiter = playback_arbiter
        self._generation = 0
        self._lifecycle_token = 0
        self._pending_key: VoiceDraftKey | None = None
        self._pending_revision = 0
        self._pending_resource: _PendingDraftTts | None = None
        self._starting: _StartingDraftTts | None = None
        self._closed = False
        self._key: VoiceDraftKey | None = None
        self._revision = 0
        self._sink: PromotablePcmSink | None = None
        self._output: DraftTtsOutput | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._text_buffer: TtsTextBuffer | SpeechTextBuffer = TtsTextBuffer(
            first_chunk_at_pause=True
        )
        self._snapshot_content = ""
        self._snapshot_speech = ""
        self._flush_offset = 0
        self._submitted_text = ""
        self._submitted_speech = ""
        self._submission_complete = False
        self._submission_task: asyncio.Task[None] | None = None
        self._promoted = False
        self._events_committed = False
        self._committed_generation: int | None = None
        self._requested_configuration: TtsConfiguration | None = None
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._live_sink_retirement_tasks: set[asyncio.Task[None]] = set()

    def attach_playback_arbiter(self, arbiter: PlaybackArbiter) -> None:
        if (
            self._closed
            or self._pending_key is not None
            or self._pending_resource is not None
            or self._key is not None
            or self._output is not None
            or self._sink is not None
            or self._event_task is not None
            or self._cleanup_tasks
            or self._live_sink_retirement_tasks
        ):
            raise RuntimeError("cannot replace playback arbiter on an active draft")
        self._playback_arbiter = arbiter

    async def start(self, key: VoiceDraftKey, *, revision: int) -> bool:
        if revision < 1 or self._closed:
            return False
        if self._committed_generation == self._generation:
            committed_key = self._key or self._pending_key
            committed_revision = (
                self._revision if self._key is not None else self._pending_revision
            )
            if committed_key is not None:
                if key != committed_key or revision != committed_revision:
                    # A speculative next turn has not acquired authority to
                    # interrupt this answer. Its LLM preparation can continue
                    # without replacing the committed answer's TTS provider.
                    self._record_event(
                        "voice_draft_tts_skipped",
                        {"reason": "committed_tts_active"},
                        key=key,
                        revision=revision,
                    )
                    return False
                starting = self._starting
                if (
                    starting is not None
                    and starting.token == self._generation
                    and not starting.ready.done()
                ):
                    ready = await asyncio.shield(starting.ready)
                    return ready and self._owns(key, revision, starting.token)
                return self._owns(key, revision, self._generation)
        self._finish_start_waiter(False)
        self._committed_generation = None
        self._lifecycle_token += 1
        token = self._lifecycle_token
        self._generation = token
        starting = _StartingDraftTts(
            key, revision, token, asyncio.get_running_loop().create_future()
        )
        self._starting = starting
        succeeded = False
        try:
            succeeded = await self._start_pending(key, revision=revision, token=token)
            return succeeded
        finally:
            if not starting.ready.done():
                starting.ready.set_result(succeeded)

    async def _start_pending(
        self, key: VoiceDraftKey, *, revision: int, token: int
    ) -> bool:
        pending_detached = self._claim_pending_resource()
        detached = self._detach_active()
        self._pending_key = key
        self._pending_revision = revision
        self._schedule_retirement(pending_detached)
        try:
            await self._await_retirement(detached)
        except asyncio.CancelledError:
            self._abandon_pending(key, revision, token)
            raise
        if not self._owns_pending(key, revision, token):
            return False
        try:
            requested_configuration = await self._load_configuration()
        except asyncio.CancelledError:
            self._abandon_pending(key, revision, token)
            raise
        except Exception:
            logger.debug("Draft TTS voice refresh missed", exc_info=True)
            self._abandon_pending(key, revision, token)
            return False
        if not self._owns_pending(key, revision, token):
            return False
        sink = PromotablePcmSink(
            self._live_sink,
            self._max_buffer_bytes,
            on_event=lambda name, metadata: self._record_event(
                name,
                metadata,
                key=key,
                revision=revision,
            ),
            playback_arbiter=self._playback_arbiter,
        )
        output = self._output_factory(sink)
        try:
            output.configure(
                model=requested_configuration.model,
                voice=requested_configuration.voice,
            )
            configure_tags = getattr(output, "set_speech_tags", None)
            if callable(configure_tags):
                configure_tags(requested_configuration.speech_tags)
            readiness_task = asyncio.create_task(
                output.start(),
                name=f"companion-draft-tts-ready-{key.turn_id}-{revision}",
            )
        except asyncio.CancelledError:
            self._abandon_pending(key, revision, token)
            self._schedule_retirement(_DetachedDraftTts(output, sink, None, None))
            raise
        except Exception:
            logger.debug("Draft TTS provider readiness missed", exc_info=True)
            self._abandon_pending(key, revision, token)
            await self._await_retirement(_DetachedDraftTts(output, sink, None, None))
            return False
        resource = _PendingDraftTts(
            token,
            key,
            revision,
            output,
            sink,
            readiness_task,
        )
        if not self._owns_pending(key, revision, token):
            readiness_task.cancel()
            self._schedule_retirement(self._pending_to_detached(resource))
            return False
        self._pending_resource = resource
        try:
            await asyncio.shield(readiness_task)
        except asyncio.CancelledError:
            claimed = self._claim_pending_resource(token=token)
            if claimed is not None:
                self._schedule_retirement(claimed)
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            return False
        except Exception:
            logger.debug("Draft TTS provider readiness missed", exc_info=True)
            claimed = self._claim_pending_resource(token=token)
            if claimed is not None:
                await self._await_retirement(claimed)
            return False
        if not self._owns_pending(key, revision, token):
            return False
        if self._pending_resource is not resource:
            return False
        self._record_event(
            "voice_draft_tts_provider_ready",
            {"revision": revision, "providerReady": True},
            key=key,
            revision=revision,
        )
        self._pending_resource = None
        self._pending_key = None
        self._pending_revision = 0
        self._key = key
        self._revision = revision
        self._sink = sink
        self._output = output
        self._text_buffer = (
            SpeechTextBuffer()
            if requested_configuration.speech_tags
            else TtsTextBuffer(first_chunk_at_pause=True)
        )
        self._snapshot_content = ""
        self._snapshot_speech = ""
        self._flush_offset = 0
        self._submitted_text = ""
        self._submitted_speech = ""
        self._submission_complete = False
        self._submission_task = None
        self._promoted = False
        self._events_committed = False
        self._requested_configuration = requested_configuration
        self._event_task = asyncio.create_task(
            self._consume_events(key, revision, token),
            name=f"companion-draft-tts-{key.turn_id}-{revision}",
        )
        self._event_task.add_done_callback(self._observe_event_task)
        return True

    async def accept_event(self, event: dict[str, object]) -> None:
        key = self._key
        if key is None:
            return
        await self._accept_event(event, key, self._revision, self._generation)

    def mark_committed(self, key: VoiceDraftKey, *, revision: int) -> bool:
        """Transfer lifecycle ownership without authorizing any public PCM.

        The server may remove the private SSE entry before provider startup
        finishes. Its empty EOF then means the public answer owns delivery;
        exact text and playback ownership are still checked by ``promote``.
        """
        generation = self._generation
        if self._closed or not (
            self._owns(key, revision, generation)
            or self._owns_pending(key, revision, generation)
        ):
            return False
        self._committed_generation = generation
        return True

    async def promote(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        flush_offset: int,
        playback_owner: PlaybackOwner | None = None,
        before_playback: Callable[[], None] | None = None,
        speech_content: str | None = None,
    ) -> DraftTtsPromotion | DraftTtsPending | None:
        generation = self._generation
        starting = self._starting
        if (
            starting is not None
            and starting.key == key
            and starting.revision == revision
            and starting.token == generation
            and not starting.ready.done()
        ):
            # Join the provider that is already connecting. Cancelling this
            # waiter must not cancel another owner of the startup request.
            await asyncio.shield(starting.ready)
        if not self._owns(key, revision, generation):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        if self._promoted:
            return None
        if isinstance(self._text_buffer, SpeechTextBuffer):
            if speech_content is None or not speech_content.startswith(
                self._snapshot_speech
            ):
                if speech_content is not None and self._snapshot_speech.startswith(
                    speech_content
                ):
                    return DraftTtsPending()
                await self._finalize_promotion_miss(key, revision, generation)
                return None
        playback_arbiter = self._playback_arbiter
        if playback_arbiter is not None and (
            playback_owner is None or not playback_arbiter.is_current(playback_owner)
        ):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        try:
            current_configuration = await self._load_configuration()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Draft TTS promotion voice refresh missed", exc_info=True)
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        if not self._owns(key, revision, generation):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        if self._promoted:
            return None
        status = self._promotion_status(
            content,
            flush_offset,
            current_configuration,
            playback_owner=playback_owner,
            allow_inflight_submission=True,
        )
        if status == "pending":
            return DraftTtsPending()
        if status == "invalid":
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        # Keep the reader alive while this existing send completes. A later
        # private snapshot may advance beyond the public prefix during the wait.
        submission_task = self._submission_task
        if submission_task is not None:
            try:
                await asyncio.shield(submission_task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                await self._finalize_promotion_miss(key, revision, generation)
                return None
            except Exception:
                await self._finalize_promotion_miss(key, revision, generation)
                return None
        if not self._owns(key, revision, generation):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        status = self._promotion_status(
            content,
            flush_offset,
            current_configuration,
            playback_owner=playback_owner,
        )
        if status == "pending":
            return DraftTtsPending()
        if status == "invalid":
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        event_task = self._event_task
        self._event_task = None
        if event_task is not None and event_task is not asyncio.current_task():
            event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
        if not self._owns(key, revision, generation):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        sink = self._sink
        output = self._output
        requested_configuration = self._requested_configuration
        if (
            sink is None
            or output is None
            or requested_configuration is None
            or (bool(self._submitted_text) and not self._submission_complete)
            or not self._snapshot_content.startswith(self._submitted_text)
            or not content.startswith(self._snapshot_content)
            or not 0 <= flush_offset <= len(content)
            or flush_offset < self._flush_offset
            or current_configuration != requested_configuration
            or output.configuration != requested_configuration
            or not self._provider_is_eligible(output, requested_configuration)
        ):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        while True:
            await self._wait_for_live_sink_retirements()
            if not self._owns(key, revision, generation):
                await self._finalize_promotion_miss(key, revision, generation)
                return None
            if self._promoted:
                return None
            try:
                current_configuration = await self._load_configuration()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Draft TTS post-retirement voice refresh missed",
                    exc_info=True,
                )
                await self._finalize_promotion_miss(key, revision, generation)
                return None
            if not self._owns(key, revision, generation):
                await self._finalize_promotion_miss(key, revision, generation)
                return None
            if self._promoted:
                return None
            if not self._live_sink_retirement_tasks:
                break
        if (
            self._sink is not sink
            or self._output is not output
            or self._requested_configuration != requested_configuration
            or current_configuration != requested_configuration
            or output.configuration != requested_configuration
            or not self._provider_is_eligible(output, requested_configuration)
            or (
                isinstance(self._text_buffer, SpeechTextBuffer)
                and (
                    speech_content is None
                    or not speech_content.startswith(self._snapshot_speech)
                    or not self._snapshot_speech.startswith(self._submitted_speech)
                    or strip_speech_tags(self._submitted_speech) != self._submitted_text
                )
            )
        ):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        public_output: DraftTtsOutput | BoundTtsOutput = output

        def attach_output() -> None:
            nonlocal public_output
            if playback_arbiter is not None and playback_owner is not None:
                public_output = playback_arbiter.bind_tts(playback_owner, output)
            if before_playback is not None:
                before_playback()

        if not await sink.promote(
            expected_end_offset=len(self._submitted_text),
            owner=playback_owner,
            before_playback=attach_output,
        ):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        # Starting playback can yield while a connecting/streaming provider
        # finishes or fails. Do not mark its unsounded prefix as adopted then.
        if (
            not self._owns(key, revision, generation)
            or self._sink is not sink
            or self._output is not output
            or output.configuration != requested_configuration
            or not self._provider_is_eligible(output, requested_configuration)
            or (
                playback_arbiter is not None
                and (
                    playback_owner is None
                    or not playback_arbiter.is_current(playback_owner)
                )
            )
        ):
            await self._finalize_promotion_miss(key, revision, generation)
            return None
        self._promoted = True
        return DraftTtsPromotion(
            public_output,
            len(self._submitted_text),
            len(self._submitted_speech)
            if isinstance(self._text_buffer, SpeechTextBuffer)
            else None,
        )

    def _promotion_status(
        self,
        content: str,
        flush_offset: int,
        current_configuration: TtsConfiguration,
        *,
        playback_owner: PlaybackOwner | None,
        allow_inflight_submission: bool = False,
    ) -> Literal["continue", "pending", "invalid"]:
        output = self._output
        configuration = self._requested_configuration
        submission = self._submission_task
        arbiter = self._playback_arbiter
        if (
            self._sink is None
            or output is None
            or configuration is None
            or current_configuration != configuration
            or output.configuration != configuration
            or (
                arbiter is not None
                and (playback_owner is None or not arbiter.is_current(playback_owner))
            )
            or not self._snapshot_content.startswith(self._submitted_text)
            or not 0 <= flush_offset <= len(content)
            or not (
                content.startswith(self._snapshot_content)
                or self._snapshot_content.startswith(content)
            )
            or (
                submission is not None
                and submission.done()
                and (submission.cancelled() or submission.exception() is not None)
            )
        ):
            return "invalid"
        if not self._provider_is_eligible(output, configuration):
            # ApiTtsOutput.append may be reconnecting the same submission. Join
            # that existing send before deciding whether its provider failed.
            if (
                allow_inflight_submission
                and submission is not None
                and not submission.done()
                and output.provider_state.error_free
            ):
                return "continue"
            return "invalid"
        if (
            len(content) < len(self._snapshot_content)
            or flush_offset < self._flush_offset
        ):
            return "pending"
        return "continue"

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        through_revision: int,
    ) -> None:
        pending_match = (
            self._pending_key == key and self._pending_revision <= through_revision
        )
        active_match = self._key == key and self._revision <= through_revision
        if not pending_match and not active_match:
            return
        self._finish_start_waiter(False, key=key, through_revision=through_revision)
        self._lifecycle_token += 1
        self._generation = self._lifecycle_token
        pending_detached = (
            self._claim_pending_resource(key=key, through_revision=through_revision)
            if pending_match
            else None
        )
        if pending_match:
            self._pending_key = None
            self._pending_revision = 0
        detached = self._detach_active() if active_match else self._empty_detached()
        await self._await_retirements(pending_detached, detached)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._finish_start_waiter(False)
            self._lifecycle_token += 1
            self._generation = self._lifecycle_token
            pending_detached = self._claim_pending_resource()
            self._pending_key = None
            self._pending_revision = 0
            detached = self._detach_active()
            self._schedule_retirement(pending_detached)
            self._schedule_retirement(detached)
        await self._drain_cleanup_tasks()

    async def wait_idle(self) -> None:
        """Wait until all detached providers and live-sink retirements settle."""
        await self._wait_for_live_sink_retirements()
        await self._drain_cleanup_tasks()

    async def _consume_events(
        self,
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        try:
            async for event in self._client.iter_voice_draft_events(
                key,
                revision=revision,
            ):
                if not self._owns(key, revision, generation):
                    return
                await self._accept_event(event, key, revision, generation)
            if (
                self._owns(key, revision, generation)
                and not self._events_committed
                and self._committed_generation != generation
            ):
                await self._retire_active()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Private draft TTS event stream missed", exc_info=True)
            if self._owns(key, revision, generation):
                await self._retire_active()

    async def _accept_event(
        self,
        event: dict[str, object],
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        if not self._owns(key, revision, generation) or self._promoted:
            self._record_event(
                "voice_draft_late_event_dropped",
                {"revision": revision, "reason": "stale_owner"},
                key=key,
                revision=revision,
            )
            return
        event_name = event.get("event")
        data = event.get("data")
        if not isinstance(data, dict):
            await self._retire_active()
            return
        data = cast(dict[str, object], data)
        if data.get("turnId") != str(key.turn_id) or data.get("revision") != revision:
            self._record_event(
                "voice_draft_late_event_dropped",
                {"revision": revision, "reason": "revision_mismatch"},
                key=key,
                revision=revision,
            )
            await self._retire_active()
            return
        if event_name == "draft.snapshot":
            state = data.get("state")
            if state not in {None, "active", "committed"}:
                await self._retire_active()
                return
            await self._accept_snapshot(data, key, revision, generation)
            if state == "committed" and self._owns(key, revision, generation):
                self._events_committed = True
        elif event_name == "draft.segment":
            await self._accept_segment(data, key, revision, generation)
        elif event_name in {"draft.canceled", "draft.failed"}:
            await self._retire_active()

    async def _accept_snapshot(
        self,
        data: dict[str, object],
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        content = data.get("content")
        start_offset = data.get("startOffset")
        end_offset = data.get("endOffset")
        flush_offset = data.get("flushOffset")
        if (
            not isinstance(content, str)
            or (start_offset is not None and start_offset != 0)
            or (end_offset is not None and end_offset != len(content))
            or isinstance(flush_offset, bool)
            or not isinstance(flush_offset, int)
            or not 0 <= flush_offset <= len(content)
            or flush_offset < self._flush_offset
            or not content.startswith(self._snapshot_content)
        ):
            await self._retire_active()
            return
        previous_content_length = len(self._snapshot_content)
        delta = content[previous_content_length:]
        if isinstance(self._text_buffer, SpeechTextBuffer):
            speech = data.get("speechContent")
            if (
                data.get("speechTags") is not True
                or not isinstance(speech, str)
                or not speech.startswith(self._snapshot_speech)
            ):
                await self._retire_active()
                return
            delta = speech[len(self._snapshot_speech) :]
            self._snapshot_speech = speech
        self._snapshot_content = content
        self._flush_offset = flush_offset
        if previous_content_length == 0 and content:
            self._record_event(
                "voice_draft_first_content",
                {"revision": revision, "charCount": len(content)},
                key=key,
                revision=revision,
            )
        if self._submitted_text:
            await self._flush_submitted_segment()
            return
        chunks = self._text_buffer.append(delta)
        chunks.extend(self._text_buffer.flush_through(flush_offset))
        if chunks:
            await self._submit_first(chunks[0], key, revision, generation)

    async def _accept_segment(
        self,
        data: dict[str, object],
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        flush_offset = data.get("flushOffset")
        if (
            isinstance(flush_offset, bool)
            or not isinstance(flush_offset, int)
            or not self._flush_offset <= flush_offset <= len(self._snapshot_content)
        ):
            await self._retire_active()
            return
        self._flush_offset = flush_offset
        if self._submitted_text:
            await self._flush_submitted_segment()
            return
        chunks = self._text_buffer.flush_through(flush_offset)
        if chunks:
            await self._submit_first(chunks[0], key, revision, generation)

    async def _submit_first(
        self,
        text: str,
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        output = self._output
        if output is None or not self._owns(key, revision, generation):
            return
        self._submitted_speech = text
        self._submitted_text = (
            strip_speech_tags(text)
            if isinstance(self._text_buffer, SpeechTextBuffer)
            else text
        )
        self._submission_complete = False
        task = asyncio.create_task(
            self._send_first(
                output,
                text,
                key,
                revision,
                generation,
                flush_at_pause=self._text_buffer.take_first_pause_flush(),
            ),
            name=f"companion-draft-tts-submit-{key.turn_id}-{revision}",
        )
        self._submission_task = task
        task.add_done_callback(self._observe_event_task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Draft TTS first segment missed", exc_info=True)
            if self._owns(key, revision, generation):
                await self._retire_active()

    async def _send_first(
        self,
        output: DraftTtsOutput,
        text: str,
        key: VoiceDraftKey,
        revision: int,
        generation: int,
        *,
        flush_at_pause: bool,
    ) -> None:
        await output.append(text)
        if self._owns(key, revision, generation):
            if flush_at_pause:
                await output.flush()
            else:
                await self._flush_submitted_segment()
            if not self._owns(key, revision, generation):
                return
            self._submission_complete = True
            self._record_event(
                "voice_draft_tts_text_submitted",
                {"revision": revision, "submittedChars": len(text)},
                key=key,
                revision=revision,
            )

    async def _flush_submitted_segment(self) -> None:
        if (
            self._output is not None
            and self._submitted_text
            and self._flush_offset >= len(self._submitted_text)
        ):
            await self._output.flush()

    def _owns(self, key: VoiceDraftKey, revision: int, generation: int) -> bool:
        return (
            self._key == key
            and self._revision == revision
            and self._generation == generation
        )

    async def _finalize_promotion_miss(
        self,
        key: VoiceDraftKey,
        revision: int,
        generation: int,
    ) -> None:
        pending_match = (
            self._lifecycle_token == generation
            and self._pending_key == key
            and self._pending_revision <= revision
        )
        active_match = (
            self._generation == generation
            and self._key == key
            and self._revision <= revision
        )
        if pending_match or active_match:
            self._finish_start_waiter(False, token=generation)
            self._lifecycle_token += 1
            self._generation = self._lifecycle_token
        pending_detached = (
            self._claim_pending_resource(token=generation) if pending_match else None
        )
        if pending_match:
            self._pending_key = None
            self._pending_revision = 0
        active_detached = (
            self._detach_active() if active_match else self._empty_detached()
        )
        await self._await_retirements(pending_detached, active_detached)
        await self._wait_for_live_sink_retirements()

    @staticmethod
    def _provider_is_eligible(
        output: DraftTtsOutput,
        requested_configuration: TtsConfiguration,
    ) -> bool:
        provider_state = output.provider_state
        return (
            provider_state.ready
            and provider_state.active
            and provider_state.error_free
            and bool(provider_state.provider)
            and provider_state.configuration == requested_configuration
        )

    def _owns_pending(
        self,
        key: VoiceDraftKey,
        revision: int,
        token: int,
    ) -> bool:
        return (
            not self._closed
            and self._lifecycle_token == token
            and self._pending_key == key
            and self._pending_revision == revision
        )

    def _abandon_pending(
        self,
        key: VoiceDraftKey,
        revision: int,
        token: int,
    ) -> None:
        if not self._owns_pending(key, revision, token):
            return
        self._finish_start_waiter(False, token=token)
        self._lifecycle_token += 1
        self._generation = self._lifecycle_token
        self._pending_key = None
        self._pending_revision = 0

    async def _retire_active(self) -> None:
        if (
            self._output is None
            and self._sink is None
            and self._event_task is None
            and self._key is None
        ):
            return
        self._lifecycle_token += 1
        self._generation = self._lifecycle_token
        detached = self._detach_active()
        await self._await_retirement(detached)

    def _detach_active(self) -> _DetachedDraftTts:
        detached = _DetachedDraftTts(
            self._output,
            self._sink,
            self._event_task,
            None,
            self._submission_task,
        )
        self._key = None
        self._revision = 0
        self._sink = None
        self._output = None
        self._event_task = None
        self._snapshot_content = ""
        self._snapshot_speech = ""
        self._flush_offset = 0
        self._submitted_text = ""
        self._submitted_speech = ""
        self._submission_complete = False
        self._submission_task = None
        self._promoted = False
        self._events_committed = False
        self._committed_generation = None
        self._requested_configuration = None
        self._text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
        task = detached.event_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        submission = detached.submission_task
        if submission is not None and not submission.done():
            submission.cancel()
        if detached.sink is not None:
            detached.sink.discard()
        return detached

    async def _retire_detached(self, detached: _DetachedDraftTts) -> None:
        submission_task = detached.submission_task
        if (
            submission_task is not None
            and submission_task is not asyncio.current_task()
        ):
            if not submission_task.done():
                submission_task.cancel()
            await asyncio.gather(submission_task, return_exceptions=True)
        readiness_task = detached.readiness_task
        if readiness_task is not None and readiness_task is not asyncio.current_task():
            if not readiness_task.done():
                readiness_task.cancel()
            await asyncio.gather(readiness_task, return_exceptions=True)
        event_task = detached.event_task
        if event_task is not None and event_task is not asyncio.current_task():
            if not event_task.done():
                event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
        if detached.sink is not None:
            await detached.sink.revoke()
        if detached.output is not None:
            try:
                await detached.output.interrupt()
            except Exception:
                logger.debug("Unable to interrupt retired draft TTS", exc_info=True)
            self._schedule_transport_close(detached.output)

    def _claim_pending_resource(
        self,
        *,
        token: int | None = None,
        key: VoiceDraftKey | None = None,
        through_revision: int | None = None,
    ) -> _DetachedDraftTts | None:
        resource = self._pending_resource
        if resource is None:
            return None
        if token is not None and resource.token != token:
            return None
        if key is not None and resource.key != key:
            return None
        if through_revision is not None and resource.revision > through_revision:
            return None
        self._pending_resource = None
        if (
            self._pending_key == resource.key
            and self._pending_revision == resource.revision
        ):
            self._pending_key = None
            self._pending_revision = 0
        resource.readiness_task.cancel()
        resource.sink.discard()
        self._finish_start_waiter(False, token=resource.token)
        return self._pending_to_detached(resource)

    def _finish_start_waiter(
        self,
        succeeded: bool,
        *,
        key: VoiceDraftKey | None = None,
        through_revision: int | None = None,
        token: int | None = None,
    ) -> None:
        starting = self._starting
        if starting is None:
            return
        if key is not None and starting.key != key:
            return
        if through_revision is not None and starting.revision > through_revision:
            return
        if token is not None and starting.token != token:
            return
        if not starting.ready.done():
            starting.ready.set_result(succeeded)

    @staticmethod
    def _pending_to_detached(resource: _PendingDraftTts) -> _DetachedDraftTts:
        return _DetachedDraftTts(
            resource.output,
            resource.sink,
            None,
            resource.readiness_task,
        )

    async def _await_retirement(
        self,
        detached: _DetachedDraftTts | None,
    ) -> None:
        task = self._schedule_retirement(detached)
        if task is not None:
            await asyncio.shield(task)

    async def _await_retirements(
        self,
        *detached_resources: _DetachedDraftTts | None,
    ) -> None:
        tasks = [
            task
            for detached in detached_resources
            if (task := self._schedule_retirement(detached)) is not None
        ]
        if tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=False,
            )

    def _schedule_retirement(
        self,
        detached: _DetachedDraftTts | None,
    ) -> asyncio.Task[None] | None:
        if detached is None or detached == self._empty_detached():
            return None
        task = asyncio.create_task(
            self._retire_detached(detached),
            name="companion-draft-tts-retire",
        )
        if detached.sink is not None and detached.sink.retirement_may_mutate_live_sink:
            self._live_sink_retirement_tasks.add(task)
        self._track_cleanup(task)
        return task

    async def _wait_for_live_sink_retirements(self) -> None:
        while self._live_sink_retirement_tasks:
            tasks = tuple(self._live_sink_retirement_tasks)
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def _drain_cleanup_tasks(self) -> None:
        while self._cleanup_tasks:
            tasks = tuple(self._cleanup_tasks)
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    @staticmethod
    def _empty_detached() -> _DetachedDraftTts:
        return _DetachedDraftTts(None, None, None, None)

    def _schedule_transport_close(self, output: DraftTtsOutput) -> None:
        close_transport = getattr(output, "close_transport", None)
        if not callable(close_transport):
            return
        task = asyncio.create_task(close_transport())
        self._track_cleanup(task)

    def _track_cleanup(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._transport_cleanup_finished)

    def _transport_cleanup_finished(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.discard(task)
        self._live_sink_retirement_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.debug("Draft TTS transport cleanup failed", exc_info=True)

    def _record_event(
        self,
        name: str,
        metadata: dict[str, object],
        *,
        key: VoiceDraftKey,
        revision: int,
    ) -> None:
        record_voice_draft_event(
            self._diagnostics,
            name,
            metadata={
                "turnId": str(key.turn_id),
                "revision": revision,
                **metadata,
            },
        )

    @staticmethod
    def _observe_event_task(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except Exception:
            return
