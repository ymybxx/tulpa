"""Long-running native voice Companion process."""

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import cast
from uuid import UUID

from fish_memory.auth.credentials import credential_store_from_settings
from fish_memory.companion.asr import (
    AsrService,
    SharedSenseVoiceDraftDecoder,
    StreamingAsr,
)
from fish_memory.companion.audio import SoundDeviceFullDuplex
from fish_memory.companion.audio_processing import (
    AudioFrameProcessor,
    IdentityAec,
    WebRtcAec,
    WebRtcVad,
)
from fish_memory.companion.binding import (
    AssistantBindingClient,
    AssistantBindingWatcher,
)
from fish_memory.companion.client import ApiCompanionClient
from fish_memory.companion.diagnostics import (
    AudioDiagnostics,
    AudioDiagnosticsConfig,
)
from fish_memory.companion.draft_tts import DraftTtsSession, PromotablePcmSink
from fish_memory.companion.livekit_audio import LiveKitProcessedPcmInput
from fish_memory.companion.livekit_vad import LiveKitVad, LiveKitVadService
from fish_memory.companion.owner_voice_gate import OwnerVoiceGate, OwnerVoiceVerifier
from fish_memory.companion.performance import (
    CompanionLatencyTracker,
    CompositeAudioDiagnostics,
    PetVoicePerformanceReporter,
)
from fish_memory.companion.playback import PlaybackArbiter, PlaybackOwner
from fish_memory.companion.processor import CompanionAudioProcessor, SmartTurnInput
from fish_memory.companion.runtime_trace import VoiceRuntimeTrace
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.shared_asr import (
    SharedAsrServer,
    configured_shared_asr_socket,
)
from fish_memory.companion.smart_turn import SmartTurnAnalyzer
from fish_memory.companion.tts import ApiTtsOutput, PcmSink, TtsConfiguration
from fish_memory.companion.vad import SileroVad, SileroVadConfig, SileroVadService
from fish_memory.companion.voice_drafts import (
    SpeculativeSilenceGate,
    VoiceDraftClient,
    VoiceDraftTurnController,
)
from fish_memory.config import Settings
from fish_memory.kws.service import KwsConnection, KwsService
from fish_memory.pet.publisher import NoopPetEventSink, PetEventPublisher, PetEventSink
from fish_memory.runtime_memory import release_unused_memory

logger = logging.getLogger(__name__)

_AUDIO_RESTART_DELAYS_SECONDS = (0.25, 0.5, 1.0, 2.0, 5.0)
_AUDIO_STABLE_RUN_SECONDS = 30.0
_VOICE_DRAFT_CAPABILITY_TIMEOUT_SECONDS = 0.5


class _UnavailableOwnerVoice:
    def score_pcm(self, pcm: bytes) -> float:
        raise RuntimeError("Owner recognition unavailable; input remains blocked")


def _default_vad_service(settings: Settings) -> SileroVadService | LiveKitVadService:
    config = SileroVadConfig.from_settings(settings)
    if getattr(settings, "companion_audio_backend", "sounddevice") == "livekit":
        return LiveKitVadService(config)
    return SileroVadService(config)


def _default_audio_backend(
    settings: Settings,
) -> tuple[
    SoundDeviceFullDuplex | LiveKitProcessedPcmInput,
    SoundDeviceFullDuplex | LiveKitProcessedPcmInput,
]:
    if getattr(settings, "companion_audio_backend", "sounddevice") == "livekit":
        if settings.companion_input_device or settings.companion_output_device:
            raise ValueError("LiveKit 实验音频后端首版仅支持系统默认输入和输出设备")
        helper_path = getattr(settings, "livekit_audio_helper", None)
        if not helper_path or not helper_path.strip():
            raise ValueError(
                "LiveKit audio backend requires FISH_MEMORY_LIVEKIT_AUDIO_HELPER"
            )
        livekit = LiveKitProcessedPcmInput(helper_path)
        return livekit, livekit
    sounddevice = SoundDeviceFullDuplex(
        block_size=settings.companion_audio_block_size,
        input_device=settings.companion_input_device,
        output_device=settings.companion_output_device,
    )
    return sounddevice, sounddevice


class CompanionRunner:
    """Connects local audio devices to the existing Fish HTTP/SSE APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        api: ApiCompanionClient | None = None,
        binding_client: AssistantBindingClient | None = None,
        kws_service: KwsService | None = None,
        asr_service: AsrService | None = None,
        audio_input: SoundDeviceFullDuplex | LiveKitProcessedPcmInput | None = None,
        audio_output: PcmSink | None = None,
        session: CompanionSession | None = None,
        voice_draft_client: VoiceDraftClient | None = None,
    ) -> None:
        self.settings = settings
        self._wake_words: tuple[str, ...] = (settings.wake_word,)
        self._playback_arbiter = PlaybackArbiter()
        api_key = credential_store_from_settings(settings).current_key()
        self._api = api or ApiCompanionClient(
            settings.assistant_api_url,
            api_key=api_key,
        )
        self._voice_draft_client = voice_draft_client or cast(
            VoiceDraftClient,
            self._api,
        )
        self._draft_tts: DraftTtsSession | None = None
        self._binding_client = binding_client or AssistantBindingClient(
            settings.assistant_api_url,
            api_key=api_key,
        )
        self._kws_service = kws_service or KwsService.from_settings(settings)
        from fish_memory.companion.asr import AsrModelConfig

        self._asr_service = asr_service or AsrService(
            AsrModelConfig.from_settings(settings)
        )
        self._vad_service = _default_vad_service(settings)
        self._owner_voice_verifier: OwnerVoiceVerifier | None = None
        self._active_vad_streams: list[SileroVad | LiveKitVad] = []
        self._diagnostics = AudioDiagnostics(
            AudioDiagnosticsConfig.from_settings(settings)
        )
        self._latency_tracker = CompanionLatencyTracker()
        self._observer = VoiceRuntimeTrace(
            CompositeAudioDiagnostics(self._diagnostics, self._latency_tracker),
            context=self._voice_trace_context,
        )
        self._performance_reporter = PetVoicePerformanceReporter(
            self._api,
            self._latency_tracker,
        )
        self._pet_events: PetEventSink = (
            PetEventPublisher(
                settings.assistant_api_url,
                api_key=api_key,
            )
            if settings.desktop_pet_enabled
            else NoopPetEventSink()
        )
        if audio_input is not None:
            self._audio_input = audio_input
            self._audio_output = audio_output or cast(PcmSink, audio_input)
        else:
            self._audio_input, default_output = _default_audio_backend(settings)
            self._audio_output = audio_output or default_output
        if session is None:
            if settings.companion_audio_block_size != 160:
                raise ValueError(
                    "FISH_MEMORY_COMPANION_AUDIO_BLOCK_SIZE must be 160 "
                    "for the 10 ms WebRTC audio frame"
                )
            tts = ApiTtsOutput(
                settings.assistant_api_url,
                sink=self._audio_output,
                sample_rate=16_000,
                api_key=api_key,
                diagnostics=self._observer,
                before_start=self._refresh_voice_config,
                facial=settings.desktop_pet_enabled,
            )
            self._tts_output: ApiTtsOutput | None = tts
            draft_tts: DraftTtsSession | None = None
            draft_buffer_ms = int(
                getattr(
                    settings,
                    "companion_voice_draft_tts_max_buffer_ms",
                    0,
                )
            )
            if settings.companion_voice_draft_enabled and draft_buffer_ms > 0:
                max_buffer_bytes = 16_000 * 2 * draft_buffer_ms // 1000

                def create_draft_output(
                    sink: PromotablePcmSink,
                ) -> ApiTtsOutput:
                    return ApiTtsOutput(
                        settings.assistant_api_url,
                        sink=sink,
                        sample_rate=16_000,
                        api_key=api_key,
                        facial=settings.desktop_pet_enabled,
                    )

                draft_tts = DraftTtsSession(
                    client=self._api,
                    live_sink=self._audio_output,
                    max_buffer_bytes=max_buffer_bytes,
                    output_factory=create_draft_output,
                    load_configuration=self._load_voice_configuration,
                    diagnostics=self._observer,
                    playback_arbiter=self._playback_arbiter,
                )
                self._draft_tts = draft_tts
            session = CompanionSession(
                self._api,
                tts,
                diagnostics=self._observer,
                performance_reporter=self._performance_reporter,
                pet_events=self._pet_events,
                draft_tts=draft_tts,
                pet_audio_sample_rate=16_000,
                home_actions_enabled=settings.home_assistant_enabled,
                home_action_timeout_seconds=settings.home_router_timeout_seconds + 3,
                playback_arbiter=self._playback_arbiter,
                agent_reply_context_ttl_seconds=(
                    settings.agent_reply_context_ttl_seconds
                ),
            )
        else:
            self._tts_output = None
            attach_playback = getattr(session, "attach_playback_arbiter", None)
            if callable(attach_playback):
                attach_playback(self._playback_arbiter)
        self._session = session
        set_playback_callback = getattr(
            self._audio_output,
            "set_playback_started_callback",
            None,
        )
        if callable(set_playback_callback):
            set_playback_callback(self._on_playback_started)
        self._voice_draft_gate: SpeculativeSilenceGate | None = None
        self._voice_draft_vad: WebRtcVad | None = None
        self._voice_draft_capture_vad: WebRtcVad | None = None
        self._voice_draft_decoder: SharedSenseVoiceDraftDecoder | None = None
        self._voice_draft_controller: VoiceDraftTurnController | None = None
        # SenseVoice owns a large ONNX Runtime arena. Retain one recognizer for
        # the Companion lifetime instead of rebuilding it after every recoverable
        # audio-device restart.
        self._authoritative_asr: StreamingAsr | None = None
        shared_asr_socket = configured_shared_asr_socket()
        self._shared_asr_server = (
            SharedAsrServer(shared_asr_socket)
            if shared_asr_socket is not None
            else None
        )
        set_rendered_audio_callback = getattr(
            self._audio_output,
            "set_rendered_audio_callback",
            None,
        )
        if settings.desktop_pet_enabled and callable(set_rendered_audio_callback):
            set_rendered_audio_callback(self._session.on_rendered_audio)
        self._stop_event = asyncio.Event()
        self._watcher: AssistantBindingWatcher | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._audio_stop: asyncio.Event | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._kws_connection: KwsConnection | None = None
        self._smart_turn_analyzer: SmartTurnAnalyzer | None = None
        self._closed = False
        self._microphone_enabled = settings.companion_microphone_enabled
        self._playback_enabled = settings.companion_playback_enabled
        self._audio_processor: CompanionAudioProcessor | None = None
        self._microphone_lock = asyncio.Lock()
        bind_microphone = getattr(self._pet_events, "set_microphone_handler", None)
        if callable(bind_microphone):
            bind_microphone(self._set_microphone_enabled)
        bind_playback = getattr(self._pet_events, "set_playback_handler", None)
        if callable(bind_playback):
            bind_playback(self._set_playback_enabled)
        bind_listen = getattr(self._pet_events, "set_listen_handler", None)
        if callable(bind_listen):
            bind_listen(self._listen_now)
        bind_new_session = getattr(self._pet_events, "set_new_session_handler", None)
        if callable(bind_new_session):
            bind_new_session(self._session.start_new_session)
        bind_stop = getattr(self._pet_events, "set_stop_listening_handler", None)
        if callable(bind_stop):
            bind_stop(self._stop_listening)
        bind_owner_voice_add = getattr(
            self._pet_events, "set_owner_voice_add_handler", None
        )
        if callable(bind_owner_voice_add):
            bind_owner_voice_add(self._add_unrecognized_owner_voice)
        self._owner_voice_gate: OwnerVoiceGate | None = None
        self._unrecognized_published: float | None = None
        self._unrecognized_expiry: asyncio.Task[None] | None = None

    async def _listen_now(self) -> bool:
        """The chat page's mic button: the same as hearing the wake word."""
        if not self._microphone_enabled:
            return False
        return await self._session.on_wake_word()

    async def _stop_listening(self) -> bool:
        """The chat page's mic button pressed while TA listens."""
        processor = self._audio_processor
        if processor is not None:
            await processor.discard_input()
        return await self._session.stop_listening()

    async def _set_playback_enabled(self, enabled: bool) -> None:
        self._playback_enabled = enabled
        apply = getattr(
            getattr(self, "_audio_output", self._audio_input),
            "set_playback_enabled",
            None,
        )
        if callable(apply):
            apply(enabled)
        logger.info(
            "Speaker output %s; microphone state unchanged",
            "enabled" if enabled else "muted",
        )

    async def _set_microphone_enabled(self, enabled: bool) -> None:
        async with self._microphone_lock:
            self._microphone_enabled = enabled
            # Gate input immediately, even while a native mute command is in flight.
            processor = self._audio_processor
            if processor is not None:
                await processor.set_microphone_enabled(enabled)
            apply = getattr(self._audio_input, "set_microphone_enabled", None)
            if callable(apply):
                apply(enabled)
            if enabled and bool(getattr(self._audio_input, "playback_only", False)):
                logger.info(
                    "Microphone input requested; reopening audio devices, "
                    "current playback will stop"
                )
            else:
                logger.info(
                    "Microphone input %s; playback remains active",
                    "enabled" if enabled else "muted",
                )

    @property
    def session(self) -> CompanionSession:
        return self._session

    def _on_playback_started(self, owner: PlaybackOwner | None) -> None:
        self._observer.record_event(
            "tts_playback_started",
            source="audio",
        )
        if owner is not None:
            self._session.on_playback_started(owner)

    def _voice_trace_context(self) -> dict[str, object]:
        audio_input = getattr(self, "_audio_input", None)
        session = getattr(self, "_session", None)
        if audio_input is None or session is None:
            return {}
        owner = self._playback_arbiter.current_owner
        return {
            "state": session.state.value,
            "audioBackend": getattr(
                self.settings, "companion_audio_backend", "sounddevice"
            ),
            "vadBackend": getattr(
                getattr(self, "_vad_service", None), "backend_name", "sherpa_silero"
            ),
            "captureStage": (
                "post_vpio"
                if getattr(audio_input, "capture_is_preprocessed", False)
                else "microphone"
            ),
            "ownerRecognitionEnabled": getattr(
                self.settings, "owner_voice_enabled", False
            ),
            "outputAttenuated": session.output_attenuated_for_speech,
            "playbackOwnerKind": owner.kind if owner is not None else None,
            "frameQueueDepth": getattr(audio_input, "frame_queue_depth", None),
            "frameQueueAgeMs": getattr(audio_input, "last_dequeued_frame_age_ms", None),
            "droppedFrames": getattr(audio_input, "dropped_frames", None),
        }

    async def run(self) -> None:
        self._observer.record_event("voice_trace_started", source="runner")
        await self._pet_events.start()
        self._pet_events.state(self._session.state.value, None)
        await self._diagnostics.start()
        await self._kws_service.start()
        await self._asr_service.start()
        await self._vad_service.start()
        # The native audio helper takes seconds to open; let it start while the
        # models load. The audio supervisor reuses the running helper.
        self._audio_prestart = asyncio.create_task(
            self._prestart_audio_backend(), name="companion-audio-prestart"
        )
        await self._wait_for_models()
        await self._load_owner_voice()
        await self._load_smart_turn()
        if self._stop_event.is_set():
            return
        await asyncio.to_thread(release_unused_memory, "companion-model-warmup")
        shared_asr_server = getattr(self, "_shared_asr_server", None)
        if shared_asr_server is not None:
            try:
                await shared_asr_server.start()
            except Exception:
                logger.exception("共享环境 ASR socket 启动失败；语音主链继续运行")
                await shared_asr_server.close()
                self._shared_asr_server = None

        self._watcher = AssistantBindingWatcher(
            self._binding_client,
            poll_interval_seconds=self.settings.assistant_poll_interval_seconds,
            bound_poll_interval_seconds=max(
                self.settings.assistant_poll_interval_seconds,
                5.0,
            ),
        )
        self._watch_task = asyncio.create_task(
            self._watcher.run(self._on_binding_change)
        )
        logger.info(
            "语音伴侣已启动，等待助手 Fish 绑定；管理页: %s/manage",
            self.settings.assistant_api_url,
        )
        try:
            await self._stop_event.wait()
        finally:
            await self.close()

    async def stop(self) -> None:
        self._stop_event.set()
        quiesce = getattr(self._session, "quiesce_binding", None)
        if callable(quiesce):
            quiesce(self._session.fish_id)

    async def close(self) -> None:
        if self._closed:
            return
        prestart = getattr(self, "_audio_prestart", None)
        if prestart is not None and not prestart.done():
            prestart.cancel()
            with suppress(asyncio.CancelledError):
                await prestart
        self._closed = True
        self._stop_event.set()
        quiesce = getattr(self._session, "quiesce_binding", None)
        if callable(quiesce):
            quiesce(self._session.fish_id)
        watcher = self._watcher
        if watcher is not None:
            watcher.close()
        watch_task = self._watch_task
        self._watch_task = None
        if watch_task is not None and watch_task is not asyncio.current_task():
            with suppress(asyncio.CancelledError):
                await watch_task
        await self._stop_audio()
        shared_asr_server = getattr(self, "_shared_asr_server", None)
        try:
            if shared_asr_server is not None:
                shared_asr_server.bind(None)
                await shared_asr_server.close()
        finally:
            self._shared_asr_server = None
            # Closing the shared server first waits for its private decode worker,
            # so no environment request can retain or re-enter the recognizer.
            self._authoritative_asr = None
        await self._close_vad_streams()
        await self._session.close()
        await self._playback_arbiter.close()
        pet_events = getattr(self, "_pet_events", None)
        if pet_events is not None:
            await pet_events.close()
        await self._performance_reporter.close()
        await self._diagnostics.close()
        await self._kws_service.close()
        await self._api.close()
        await self._binding_client.close()

    async def _wait_for_models(self) -> None:
        last_log = 0.0
        while not self._stop_event.is_set():
            kws_status = self._kws_service.status()
            asr_status = self._asr_service.status()
            vad_status = self._vad_service.status()
            if (
                bool(kws_status["available"])
                and bool(asr_status["available"])
                and bool(vad_status["available"])
            ):
                return
            errors = [
                str(status["error"])
                for status in (kws_status, asr_status, vad_status)
                if status.get("error")
            ]
            if errors:
                raise RuntimeError("; ".join(errors))
            now = asyncio.get_running_loop().time()
            if now - last_log >= 1.0:
                logger.info(
                    "本地语音模型准备中: KWS %.1f%%, ASR %.1f%%, VAD %.1f%%",
                    _percent(kws_status),
                    _percent(asr_status),
                    _percent(vad_status),
                )
                last_log = now
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
            except TimeoutError:
                pass

    async def _on_binding_change(self, fish_id: UUID | None) -> None:
        previous_fish_id = self._session.fish_id
        logger.info(
            "助手 Fish 绑定已变更: %s -> %s",
            previous_fish_id,
            fish_id,
        )
        if previous_fish_id != fish_id:
            quiesce = getattr(self._session, "quiesce_binding", None)
            if callable(quiesce):
                quiesce(previous_fish_id)
        await self._stop_audio()
        await self._session.set_binding(fish_id)
        if fish_id is None:
            logger.warning("当前没有绑定助手 Fish，请打开管理页选择一个助手 Fish")
            return

        if self._tts_output is not None:
            await self._refresh_voice_config()

        self._audio_stop = asyncio.Event()
        self._audio_task = asyncio.create_task(
            self._supervise_audio(fish_id, self._audio_stop),
            name=f"companion-audio-{fish_id}",
        )

    def _create_audio_processor(
        self,
        asr_stream: StreamingAsr,
        frame_processor: AudioFrameProcessor,
        interruption_vad: SileroVad | LiveKitVad,
        *,
        kws_connection: KwsConnection | None = None,
    ) -> CompanionAudioProcessor:
        kws = kws_connection or self._kws_connection
        if kws is None:
            raise RuntimeError("KWS 流未初始化")
        notify_voice_activity = getattr(
            self._audio_input,
            "notify_voice_activity",
            None,
        )
        return CompanionAudioProcessor(
            self._session,
            kws,
            asr_stream,
            frame_processor=frame_processor,
            interruption_vad=interruption_vad,
            defer_interruption_output_until_near_end=bool(
                getattr(self._audio_input, "capture_is_preprocessed", False)
            ),
            # Silero already confirms the minimum speech duration. Keep no
            # second two-frame debounce in front of it.
            speech_start_frames=1,
            speech_pre_roll_frames=self._vad_service.config.speech_pre_roll_frames,
            speech_confirmation_frames=(
                self._vad_service.config.speech_confirmation_frames
            ),
            # Silero already applies min_silence_duration internally. Once it
            # reports false, close the outer turn immediately instead of
            # waiting through the same silence window a second time.
            speech_end_frames=1,
            utterance_commit_delay_seconds=(
                self.settings.companion_utterance_commit_delay_ms / 1000
            ),
            smart_turn=cast(SmartTurnInput | None, self._smart_turn_analyzer),
            smart_turn_reopen_seconds=(
                self.settings.companion_smart_turn_reopen_ms / 1000
            ),
            smart_turn_timeout_seconds=(
                self.settings.companion_smart_turn_timeout_ms / 1000
            ),
            early_interrupt_snapshot_seconds=(
                self.settings.companion_early_interrupt_snapshot_ms / 1000
            ),
            early_interrupt_min_chars=(
                self.settings.companion_early_interrupt_min_chars
            ),
            single_char_min_speech_ms=(
                self.settings.companion_single_char_min_speech_ms
            ),
            vad_trailing_silence_ms=round(
                self._vad_service.config.min_silence_duration * 1000
            ),
            voice_draft_vad=getattr(self, "_voice_draft_vad", None),
            voice_draft_gate=getattr(self, "_voice_draft_gate", None),
            voice_draft_controller=getattr(self, "_voice_draft_controller", None),
            voice_draft_capture_vad=getattr(self, "_voice_draft_capture_vad", None),
            voice_draft_max_pcm_bytes=round(
                self._vad_service.config.max_speech_duration * 16_000 * 2
            ),
            diagnostics=self._observer,
            dropped_frames=lambda: self._audio_input.dropped_frames,
            frame_queue_age_ms=(lambda: self._audio_input.last_dequeued_frame_age_ms),
            frame_queue_depth=lambda: self._audio_input.frame_queue_depth,
            owner_voice_gate=self._create_owner_voice_gate(),
            notify_voice_activity=(
                notify_voice_activity if callable(notify_voice_activity) else None
            ),
        )

    async def _load_owner_voice(self) -> None:
        if not self.settings.owner_voice_enabled:
            return
        from fish_memory.companion.owner_voice import OwnerVoiceService

        try:
            self._owner_voice_verifier = await asyncio.to_thread(
                OwnerVoiceService(Path(self.settings.owner_voice_dir)).load_verifier,
                audio_backend=self.settings.companion_audio_backend,
                threshold=self.settings.owner_voice_threshold,
            )
            logger.info("主人识别已启用：本地 CAM++；未知声音不会进入对话")
        except Exception as error:
            # Never silently turn an explicit 'only my voice' choice into
            # unrestricted listening. Chat/API and normal output remain live.
            self._owner_voice_verifier = _UnavailableOwnerVoice()
            logger.warning(
                "主人识别不可用（%s）；语音输入保持阻止，请重新录入或关闭主人识别",
                type(error).__name__,
            )

    def _create_owner_voice_gate(self) -> OwnerVoiceGate | None:
        if not getattr(self.settings, "owner_voice_enabled", False):
            self._owner_voice_gate = None
            return None
        verifier = self._owner_voice_verifier or _UnavailableOwnerVoice()
        self._owner_voice_gate = OwnerVoiceGate(
            verifier,
            threshold=self.settings.owner_voice_threshold,
            on_unrecognized=self._publish_unrecognized_owner_voice,
        )
        return self._owner_voice_gate

    def _publish_unrecognized_owner_voice(
        self, seconds: float, **extra: object
    ) -> None:
        """Tell the desktop how much just-missed speech the owner could add."""
        seconds = round(seconds, 1)
        if seconds == self._unrecognized_published and not extra:
            return
        self._unrecognized_published = seconds
        publish = getattr(self._pet_events, "owner_voice_unrecognized", None)
        if callable(publish):
            publish({"seconds": seconds, **extra})
        if seconds > 0 and (
            self._unrecognized_expiry is None or self._unrecognized_expiry.done()
        ):
            self._unrecognized_expiry = asyncio.get_running_loop().create_task(
                self._watch_unrecognized_expiry()
            )

    async def _watch_unrecognized_expiry(self) -> None:
        """Kept sentences are forgotten after a few minutes even in silence, so
        the offer to add them must not outlive them."""
        while True:
            await asyncio.sleep(15)
            gate = self._owner_voice_gate
            seconds = gate.unrecognized_seconds() if gate is not None else 0.0
            self._publish_unrecognized_owner_voice(seconds)
            if seconds <= 0:
                return

    async def _add_unrecognized_owner_voice(self) -> None:
        """The owner confirmed the missed sentences were theirs: learn them."""
        from fish_memory.companion.owner_voice import (
            MIN_ADDED_SPEECH_SECONDS,
            OwnerVoiceService,
        )

        gate = self._owner_voice_gate
        if gate is None or gate.unrecognized_seconds() < MIN_ADDED_SPEECH_SECONDS:
            self._publish_unrecognized_owner_voice(
                gate.unrecognized_seconds() if gate else 0.0,
                error="没认出的话还不够，再说几句再加",
            )
            return
        speech, heard = gate.take_unrecognized()
        service = OwnerVoiceService(Path(self.settings.owner_voice_dir))
        try:
            status = await asyncio.to_thread(
                service.add_speech,
                speech,
                audio_backend=self.settings.companion_audio_backend,
                heard=heard,
            )
            verifier = await asyncio.to_thread(
                service.load_verifier,
                audio_backend=self.settings.companion_audio_backend,
                threshold=self.settings.owner_voice_threshold,
            )
        except Exception as error:
            logger.warning("加入主人声纹失败（%s）", type(error).__name__)
            self._publish_unrecognized_owner_voice(
                0.0, error=str(error) if str(error) else "加入声纹没有成功"
            )
            return
        finally:
            del speech, heard
        self._owner_voice_verifier = verifier
        gate.replace_verifier(verifier)
        logger.info("已把没认出的话加入主人声纹：共 %s 段", status["enrollmentCount"])
        self._publish_unrecognized_owner_voice(
            0.0, enrollmentCount=int(cast(int, status["enrollmentCount"]))
        )

    async def _open_audio_processor(
        self,
    ) -> tuple[CompanionAudioProcessor, KwsConnection]:
        """Create one isolated set of native streams for an audio run."""
        await self._close_vad_streams()
        self._wake_words = await self._load_wake_words()
        connection = await self._kws_service.open_stream(self._wake_words)
        processor: CompanionAudioProcessor | None = None
        try:
            asr_stream = await self._open_asr_stream()
            vad_stream = await self._open_vad_stream()
            self._active_vad_streams.append(vad_stream)
            interruption_vad = await self._open_vad_stream()
            self._active_vad_streams.append(interruption_vad)
            capture_is_preprocessed = bool(
                getattr(self._audio_input, "capture_is_preprocessed", False)
            )
            aec = (
                IdentityAec(stream_delay_ms=self.settings.companion_aec_delay_ms)
                if capture_is_preprocessed
                else WebRtcAec(
                    stream_delay_ms=self.settings.companion_aec_delay_ms,
                    delay_mode=self.settings.companion_aec_delay_mode,
                )
            )
            frame_processor = AudioFrameProcessor(aec, vad_stream)
            self._kws_connection = connection
            processor = self._create_audio_processor(
                asr_stream,
                frame_processor,
                interruption_vad,
                kws_connection=connection,
            )
            processor.set_stream_delay(self.settings.companion_aec_delay_ms)
            async with self._microphone_lock:
                if not self._microphone_enabled:
                    await processor.set_microphone_enabled(False)
                self._audio_processor = processor
        except BaseException:
            if processor is not None:
                with suppress(Exception):
                    await processor.close()
            await self._close_vad_streams()
            if self._kws_connection is connection:
                self._kws_connection = None
            with suppress(Exception):
                await asyncio.to_thread(connection.close)
            raise

        processing_label = (
            "LiveKit 原生预处理"
            if bool(getattr(self._audio_input, "capture_is_preprocessed", False))
            else "WebRTC AEC"
        )
        processing_label += (
            " + LiveKit 官方本地 Silero VAD"
            if getattr(self._vad_service, "backend_name", None) == "livekit_silero"
            else " + Silero VAD"
        )
        logger.info(
            "%s 已启用: sample_rate=16000, frame_ms=10, "
            "delay_mode=%s, delay_ms=%s, threshold=%.2f, min_silence_ms=%s, "
            "min_speech_ms=%s",
            processing_label,
            self.settings.companion_aec_delay_mode,
            self.settings.companion_aec_delay_ms,
            self._vad_service.config.threshold,
            round(self._vad_service.config.min_silence_duration * 1000),
            round(self._vad_service.config.min_speech_duration * 1000),
        )
        return processor, connection

    async def _open_vad_stream(self) -> SileroVad | LiveKitVad:
        service = self._vad_service
        if getattr(service, "backend_name", None) != "livekit_silero":
            return service.open_stream()
        # Plugin registration must happen on the main thread. Native model
        # construction may take longer, so keep it off the Companion loop.
        await service.start()
        opening = asyncio.create_task(asyncio.to_thread(service.open_stream))
        try:
            return await asyncio.shield(opening)
        except asyncio.CancelledError:
            # to_thread cannot cancel construction. Wait for its bounded result
            # and retire the unclaimed worker instead of leaving an orphan.
            with suppress(Exception):
                stream = await opening
                await asyncio.to_thread(cast(LiveKitVad, stream).close)
            raise

    async def _close_vad_streams(self) -> None:
        # Each official VAD stream owns a local async worker. Retire it only
        # after the processor's pending finalization/reset work is stopped.
        streams = tuple(getattr(self, "_active_vad_streams", ()))
        self._active_vad_streams = []
        for stream in streams:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except Exception:
                    logger.exception("VAD 流关闭失败，继续回收其它语音资源")

    async def _open_asr_stream(self) -> StreamingAsr:
        """Prepare the retained SenseVoice recognizer for one audio run."""
        await self._close_voice_drafts()
        if not bool(getattr(self.settings, "companion_voice_draft_enabled", False)):
            return await self._prepare_authoritative_asr()
        authoritative_task = asyncio.create_task(
            self._prepare_authoritative_asr(),
            name="companion-authoritative-asr-prepare",
        )
        try:
            supported = await asyncio.wait_for(
                self._api.voice_draft_capability(),
                timeout=_VOICE_DRAFT_CAPABILITY_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            authoritative_task.cancel()
            await asyncio.gather(authoritative_task, return_exceptions=True)
            raise
        except Exception:
            logger.warning(
                "语音草稿能力探测失败，本次音频运行保持普通 /ask 路径",
                exc_info=True,
            )
            return await authoritative_task
        if not supported:
            logger.info("API 未启用语音草稿，本次音频运行保持普通 /ask 路径")
            return await authoritative_task

        authoritative = await authoritative_task
        try:
            decoder = SharedSenseVoiceDraftDecoder(authoritative)
        except Exception as error:
            logger.warning(
                "语音草稿 ASR 共享失败，继续使用权威 ASR: %s",
                error,
            )
            return authoritative
        logger.info("语音草稿复用权威 SenseVoice ASR；未加载第二份识别模型")
        try:
            draft_vad = WebRtcVad(mode=2)
            capture_vad = WebRtcVad(mode=2)
        except Exception:
            logger.exception("语音草稿 WebRTC VAD 加载失败，继续使用权威 ASR/VAD 路径")
            return authoritative
        candidate_silence_ms = int(
            getattr(
                self.settings,
                "companion_voice_draft_candidate_silence_ms",
                100,
            )
        )
        silence_frames = max(1, (candidate_silence_ms + 9) // 10)
        self._voice_draft_gate = SpeculativeSilenceGate(
            silence_frames=silence_frames,
            prepare_silence_frames=1,
        )
        self._voice_draft_vad = draft_vad
        self._voice_draft_capture_vad = capture_vad
        self._voice_draft_decoder = decoder
        observer = getattr(self, "_observer", None)
        if observer is None:
            self._voice_draft_controller = VoiceDraftTurnController(
                client=self._voice_draft_client,
                decoder=decoder,
                before_context=getattr(
                    getattr(self, "_session", None),
                    "settle_playback_before_context",
                    None,
                ),
                draft_tts=getattr(self, "_draft_tts", None),
            )
        else:
            self._voice_draft_controller = VoiceDraftTurnController(
                client=self._voice_draft_client,
                decoder=decoder,
                before_context=getattr(
                    getattr(self, "_session", None),
                    "settle_playback_before_context",
                    None,
                ),
                draft_tts=getattr(self, "_draft_tts", None),
                diagnostics=observer,
            )
        return authoritative

    async def _prepare_authoritative_asr(self) -> StreamingAsr:
        """Open SenseVoice once, then clear only per-run utterance state.

        ``StreamingAsr.reset`` touches the PCM buffer, not the native recognizer,
        so it is safe while the shared environment worker is decoding an immutable
        snapshot. Native decode admission remains serialized inside StreamingAsr.
        """
        authoritative = getattr(self, "_authoritative_asr", None)
        if authoritative is None:
            opening = asyncio.create_task(
                asyncio.to_thread(self._asr_service.open_stream),
                name="companion-authoritative-asr-open",
            )
            try:
                authoritative = await asyncio.shield(opening)
            except asyncio.CancelledError:
                # Native construction cannot be cancelled. Retain its completed
                # model for the next binding/run instead of orphaning a large ORT
                # allocation and opening a second copy on recovery.
                result = await asyncio.gather(opening, return_exceptions=True)
                if result and not isinstance(result[0], BaseException):
                    self._retain_authoritative_asr(cast(StreamingAsr, result[0]))
                raise
            self._retain_authoritative_asr(authoritative)

        # Processor.close fences old callbacks but deliberately does not own this
        # persistent recognizer. Start every run from an empty utterance buffer.
        await asyncio.to_thread(authoritative.reset)
        return authoritative

    def _retain_authoritative_asr(self, authoritative: StreamingAsr) -> None:
        self._authoritative_asr = authoritative
        shared_asr_server = getattr(self, "_shared_asr_server", None)
        if shared_asr_server is not None:
            shared_asr_server.bind(authoritative)

    async def _close_voice_drafts(self) -> None:
        gate = getattr(self, "_voice_draft_gate", None)
        controller = getattr(self, "_voice_draft_controller", None)
        self._voice_draft_gate = None
        self._voice_draft_vad = None
        self._voice_draft_capture_vad = None
        self._voice_draft_decoder = None
        self._voice_draft_controller = None
        if gate is not None:
            gate.close_turn()
        if controller is not None:
            await controller.close()

    async def _load_smart_turn(self) -> None:
        self._smart_turn_analyzer = None
        if not self.settings.companion_smart_turn_enabled:
            logger.info("Smart Turn 已禁用，使用 Silero VAD 端点")
            return
        try:
            analyzer = await asyncio.to_thread(
                SmartTurnAnalyzer,
                model_path=self.settings.companion_smart_turn_model_path,
                threshold=self.settings.companion_smart_turn_threshold,
                num_threads=self.settings.companion_smart_turn_num_threads,
            )
            await asyncio.to_thread(analyzer.warm_up)
        except Exception:
            logger.exception("Smart Turn 加载失败，使用 Silero VAD 端点")
            return
        self._smart_turn_analyzer = analyzer
        logger.info(
            "Smart Turn 已启用: threshold=%.2f, reopen_ms=%s, "
            "feature_extractor=numpy_whisper",
            self.settings.companion_smart_turn_threshold,
            self.settings.companion_smart_turn_reopen_ms,
        )

    async def _refresh_voice_config(self) -> None:
        fish_id = self._session.fish_id
        if fish_id is None or self._tts_output is None:
            return
        try:
            voice = await self._load_voice_configuration()
            self._tts_output.configure(model=voice.model, voice=voice.voice)
            configure_tags = getattr(
                self._tts_output, "set_speech_tag_capability", None
            )
            if callable(configure_tags):
                configure_tags(voice.speech_tags)
            logger.debug(
                "助手 Fish 音色已刷新: model=%s, voice=%s",
                voice.model,
                voice.voice,
            )
        except Exception:
            # A temporary local API failure must not make an otherwise healthy
            # voice turn silent; retain the last successfully loaded voice.
            logger.exception("无法刷新 Fish %s 的音色配置，继续使用当前音色", fish_id)

    async def _load_wake_words(self) -> tuple[str, ...]:
        """The bound Fish's wake words, or the app-wide one if it has none yet."""
        fish_id = getattr(getattr(self, "_session", None), "fish_id", None)
        fallback = (self.settings.wake_word,)
        get_wake_words = getattr(getattr(self, "_api", None), "get_wake_words", None)
        if fish_id is None or get_wake_words is None:
            return fallback
        try:
            return tuple(await get_wake_words(fish_id))
        except Exception:
            logger.exception("无法读取 Fish %s 的唤醒词，先用默认唤醒词", fish_id)
            return fallback

    async def _load_voice_configuration(self) -> TtsConfiguration:
        fish_id = self._session.fish_id
        if fish_id is None:
            raise RuntimeError("voice configuration requires a bound Fish")
        voice = await self._api.get_voice_config(fish_id)
        return TtsConfiguration(
            voice.model, voice.voice, 16_000, getattr(voice, "speech_tags", False)
        )

    async def _run_audio(
        self,
        fish_id: UUID,
        processor: CompanionAudioProcessor,
        stop_event: asyncio.Event,
    ) -> None:
        stream_ready = False

        async def process_frame(capture_pcm: bytes, render_pcm: bytes) -> None:
            nonlocal stream_ready
            playback_only = bool(getattr(self._audio_input, "playback_only", False))
            if not playback_only:
                await processor.process_frame(capture_pcm, render_pcm)
            if stream_ready:
                return
            stream_ready = True
            if playback_only:
                logger.info(
                    "已绑定助手 Fish %s，唤醒词暂停；麦克风已关闭，仅播放就绪",
                    fish_id,
                )
                return
            logger.info(
                "已绑定助手 Fish %s，唤醒词为“%s”",
                fish_id,
                "”“".join(getattr(self, "_wake_words", (self.settings.wake_word,))),
            )

        try:
            if (
                not bool(getattr(self._audio_input, "capture_is_preprocessed", False))
                and self.settings.companion_aec_delay_mode == "auto"
            ):
                stream_delay_ms = self._audio_input.stream_delay_ms
                if stream_delay_ms is not None:
                    processor.set_stream_delay(stream_delay_ms)
                    logger.info(
                        "AEC 已使用声卡输入+输出延迟: %sms",
                        stream_delay_ms,
                    )
            await self._audio_input.run(
                process_frame,
                stop_event,
                start_if_needed=False,
            )
        finally:
            try:
                await processor.close()
            finally:
                if self._audio_processor is processor:
                    self._audio_processor = None
                await self._close_vad_streams()

    async def _prestart_audio_backend(self) -> None:
        try:
            await self._start_audio_backend()
        except Exception:
            # Bound audio starts it again and reports failures there.
            logger.info("音频服务预启动未成功，绑定助手后会再试", exc_info=True)

    async def _start_audio_backend(self) -> None:
        """Validate capture/playback before allocating per-run native models."""
        await self._set_playback_enabled(self._playback_enabled)
        apply = getattr(self._audio_input, "set_microphone_enabled", None)
        if callable(apply):
            apply(self._microphone_enabled)
        await self._audio_input.start()
        audio_output = cast(
            PcmSink,
            getattr(self, "_audio_output", self._audio_input),
        )
        if audio_output is not self._audio_input:
            await audio_output.start()

    async def _supervise_audio(
        self,
        fish_id: UUID,
        stop_event: asyncio.Event,
    ) -> None:
        """Keep a bound Fish listening across recoverable native stream failures."""
        failures = 0
        while not stop_event.is_set() and not self._stop_event.is_set():
            processor: CompanionAudioProcessor | None = None
            connection: KwsConnection | None = None
            started_at = asyncio.get_running_loop().time()
            try:
                await self._start_audio_backend()
                processor, connection = await self._open_audio_processor()
                if stop_event.is_set() or self._stop_event.is_set():
                    return
                if failures and not await self._session.resume_after_audio_recovery(
                    fish_id
                ):
                    return
                await self._run_audio(fish_id, processor, stop_event)
                if stop_event.is_set() or self._stop_event.is_set():
                    return
                raise RuntimeError("本地麦克风音频流意外结束")
            except asyncio.CancelledError:
                raise
            except Exception:
                try:
                    # The failed stream may have stopped halfway through a
                    # listening or speaking turn. Finish that turn before a
                    # fresh ASR/VAD state starts, while retaining the binding.
                    recovered = await self._session.reset_for_audio_recovery(fish_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "音频流异常后无法重置 Fish %s 语音轮次，仍继续重建",
                        fish_id,
                    )
                    if self._session.fish_id != fish_id:
                        return
                else:
                    if not recovered:
                        return
                run_seconds = asyncio.get_running_loop().time() - started_at
                if run_seconds >= _AUDIO_STABLE_RUN_SECONDS:
                    failures = 0
                failures += 1
                delay = _AUDIO_RESTART_DELAYS_SECONDS[
                    min(failures - 1, len(_AUDIO_RESTART_DELAYS_SECONDS) - 1)
                ]
                logger.exception(
                    "本地麦克风音频流异常，Fish %s 保持绑定，%.2f 秒后将完整重建"
                    " KWS/ASR/VAD/AEC/声卡流（第 %s 次）",
                    fish_id,
                    delay,
                    failures,
                )
            finally:
                await self._close_voice_drafts()
                await self._close_vad_streams()
                if connection is not None:
                    if self._kws_connection is connection:
                        self._kws_connection = None
                    try:
                        await asyncio.to_thread(connection.close)
                    except Exception:
                        logger.exception("KWS 流释放失败，继续重建音频链路")
                # SoundDeviceFullDuplex.run() normally closes the PortAudio
                # stream itself. This second idempotent close also covers a
                # partial failure in start() before run() takes ownership.
                try:
                    await self._audio_input.close()
                except Exception:
                    logger.exception("声卡流释放失败，继续重建音频链路")
                audio_output = cast(
                    PcmSink,
                    getattr(self, "_audio_output", self._audio_input),
                )
                if audio_output is not self._audio_input:
                    try:
                        await audio_output.stop()
                    except Exception:
                        logger.exception("独立播放流释放失败，继续重建音频链路")

            if await self._wait_for_audio_restart(stop_event, delay):
                return

    async def _wait_for_audio_restart(
        self,
        stop_event: asyncio.Event,
        delay: float,
    ) -> bool:
        """Return true when shutdown wins the bounded recovery backoff."""
        if stop_event.is_set() or self._stop_event.is_set():
            return True
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except TimeoutError:
            return self._stop_event.is_set()
        return True

    async def _stop_audio(self) -> None:
        if self._audio_stop is not None:
            self._audio_stop.set()
        task = self._audio_task
        self._audio_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._audio_stop = None
        # The live task owns processor.close(), which may still observe a final
        # START while cancellation unwinds. Let it retire local ownership before
        # closing the controller shared with the supervisor.
        await self._close_voice_drafts()
        connection = self._kws_connection
        self._kws_connection = None
        if connection is not None:
            await asyncio.to_thread(connection.close)


def _percent(status: dict[str, object]) -> float:
    value = status.get("downloadPercent")
    return float(value) if isinstance(value, (int, float)) else 0.0
