import asyncio
import json
import logging
import threading
import time
import wave
from contextlib import suppress
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

import httpx
from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from fish_memory import failures
from fish_memory.config import Settings
from fish_memory.tts.catalog import (
    find_tts_model,
    get_local_tts_catalog,
    get_tts_catalog,
)
from fish_memory.tts.custom_voices import (
    CustomVoice,
    CustomVoiceList,
    CustomVoiceProviderError,
    DashScopeVoiceClient,
)
from fish_memory.tts.facial import SpeechFacialPlan, create_speech_facial_plan
from fish_memory.tts.local import (
    LocalTtsModelConfig,
    LocalTtsService,
    default_tts_provider,
    model_provider,
)
from fish_memory.tts.models import TtsAppendRequest, TtsRequest, TtsStartRequest
from fish_memory.tts.provider import (
    DEFAULT_DASHSCOPE_WEBSOCKET_URL,
    DashScopeTtsSession,
    TtsServerEvent,
    build_cancel_task,
    build_continue_task,
    build_finish_task,
    build_run_task,
    parse_server_event,
)
from fish_memory.tts.speech_tags import supports_speech_tags
from fish_memory.tts.tagged_stream import run_tagged_tts
from fish_memory.tts.voice_config import FishVoiceConfig, VoiceKind, is_supported_voice

tts_router = APIRouter(prefix="/v1/tts", tags=["public", "tts"])
tts_internal_router = APIRouter(
    prefix="/internal/tts",
    tags=["internal", "tts"],
    include_in_schema=False,
)
logger = logging.getLogger(__name__)

TtsConnectionKey = tuple[str, str, str, int]
TTS_CANCEL_TIMEOUT_SECONDS = 1.0
# DashScope closes a duplex task when the interval between continue-task
# messages exceeds 23 seconds. Finish an idle task before that limit, then
# wait for the next real text before starting the next task on the same socket.
TTS_IDLE_ROTATION_INTERVAL_SECONDS = 10.0


if TYPE_CHECKING:
    # The proxy forwards everything it does not override to the wrapped socket,
    # so it is passed wherever a WebSocket is expected.
    _SocketProxyBase = WebSocket
else:
    _SocketProxyBase = object


class _SerializedTtsSocket(_SocketProxyBase):
    """Keep audio metadata and its binary PCM atomic with late facial events."""

    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket
        self._send_lock = asyncio.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._websocket, name)

    async def send_json(self, data: Any, mode: str = "text") -> None:
        async with self._send_lock:
            if mode == "text":
                await self._websocket.send_json(data)
            else:
                await self._websocket.send_json(data, mode=mode)

    async def send_packet(self, metadata: dict[str, Any], pcm: bytes) -> None:
        async with self._send_lock:
            await self._websocket.send_json(metadata)
            await self._websocket.send_bytes(pcm)


def _serialized(websocket: Any) -> _SerializedTtsSocket:
    return (
        websocket
        if isinstance(websocket, _SerializedTtsSocket)
        else _SerializedTtsSocket(websocket)
    )


@tts_router.get("/models")
async def tts_models(
    request: Request,
    provider: Literal["local", "dashscope"] | None = Query(default=None),
) -> dict[str, Any]:
    """One provider's voices (the default unless asked) and whether it can speak."""
    settings = getattr(request.app.state, "settings", None)
    if provider is None:
        provider = default_tts_provider(settings) if settings is not None else "local"
    catalog = get_local_tts_catalog() if provider == "local" else get_tts_catalog()
    return {
        "provider": provider,
        "ready": provider == "local"
        or getattr(settings, "tts_api_key", None) is not None,
        "models": [model.model_dump() for model in catalog],
    }


def _custom_voice_client(request: Request) -> DashScopeVoiceClient:
    configured = getattr(request.app.state, "custom_voice_client", None)
    if configured is not None:
        return configured
    settings = getattr(request.app.state, "settings", None)
    if settings is None or getattr(settings, "tts_api_key", None) is None:
        raise RuntimeError("FISH_MEMORY_TTS_API_KEY is not configured")
    client = DashScopeVoiceClient(
        api_key=settings.tts_api_key.get_secret_value(),
        customization_url=getattr(
            settings,
            "tts_customization_url",
            "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization",
        ),
        upload_url=getattr(
            settings,
            "tts_upload_url",
            "https://dashscope.aliyuncs.com/api/v1/uploads",
        ),
    )
    request.app.state.custom_voice_client = client
    return client


@tts_internal_router.get(
    "/custom-voices",
    response_model=CustomVoiceList,
    response_model_by_alias=True,
)
async def list_custom_voices(request: Request) -> CustomVoiceList | JSONResponse:
    try:
        return CustomVoiceList(items=await _custom_voice_client(request).list_voices())
    except CustomVoiceProviderError as error:
        logger.warning(
            "DashScope rejected custom voice list request: "
            "code=%s request_id=%s message=%s",
            error.provider_code,
            error.request_id,
            error.provider_message,
        )
        return _custom_voice_provider_error(error)
    except RuntimeError as error:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "tts_not_configured", "message": str(error)}},
        )
    except httpx.HTTPError:
        logger.exception("Unable to list DashScope custom voices")
        return _custom_voice_provider_error()


@tts_internal_router.post(
    "/custom-voices",
    response_model=CustomVoice,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_custom_voice(
    request: Request,
    target_model: str = Query(alias="targetModel"),
) -> CustomVoice | JSONResponse:
    audio = await request.body()
    audio_error = _voice_audio_error(audio)
    if audio_error is not None:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "invalid_voice_audio",
                    "message": audio_error,
                }
            },
        )
    model = find_tts_model(target_model)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "unsupported_voice_model",
                    "message": "该模型不支持当前音色复刻流程",
                }
            },
        )
    try:
        return await _custom_voice_client(request).create_voice(
            audio,
            target_model=target_model,
        )
    except CustomVoiceProviderError as error:
        logger.warning(
            "DashScope rejected custom voice creation: "
            "code=%s request_id=%s message=%s",
            error.provider_code,
            error.request_id,
            error.provider_message,
        )
        return _custom_voice_provider_error(error)
    except RuntimeError as error:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "tts_not_configured", "message": str(error)}},
        )
    except httpx.HTTPError:
        logger.exception("Unable to create DashScope custom voice")
        return _custom_voice_provider_error()


@tts_internal_router.delete(
    "/custom-voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_custom_voice(
    voice_id: str,
    request: Request,
    force: bool = Query(default=False),
) -> Response:
    fish_store = request.app.state.fish_store
    reference_count = await fish_store.count_voice_references(voice_id)
    if reference_count > 0 and not force:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "voice_in_use",
                    "message": "该音色仍被 Fish 使用",
                    "details": {"fishCount": reference_count},
                }
            },
        )
    try:
        await _custom_voice_client(request).delete_voice(voice_id)
    except CustomVoiceProviderError as error:
        logger.warning(
            "DashScope rejected custom voice deletion: "
            "code=%s request_id=%s message=%s",
            error.provider_code,
            error.request_id,
            error.provider_message,
        )
        return _custom_voice_provider_error(error)
    except RuntimeError as error:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "tts_not_configured", "message": str(error)}},
        )
    except httpx.HTTPError:
        logger.exception("Unable to delete DashScope custom voice")
        return _custom_voice_provider_error()
    await fish_store.clear_voice_references(voice_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _custom_voice_provider_error(
    error: CustomVoiceProviderError | None = None,
) -> JSONResponse:
    if error is not None:
        details = {"providerCode": error.provider_code}
        if error.request_id is not None:
            details["requestId"] = error.request_id
        return JSONResponse(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
                if 400 <= error.status_code < 500
                else status.HTTP_502_BAD_GATEWAY
            ),
            content={
                "error": {
                    "code": "custom_voice_rejected",
                    "message": f"百炼拒绝创建音色：{error.provider_message}",
                    "details": details,
                }
            },
        )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": {
                "code": "custom_voice_provider_error",
                "message": "百炼音色服务暂时不可用，请稍后重试",
            }
        },
    )


def _voice_audio_error(audio: bytes) -> str | None:
    if len(audio) > 10 * 1024 * 1024:
        return "录音不能超过 10 MB"
    try:
        with wave.open(BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
    except (EOFError, wave.Error):
        return "录音必须是有效的 PCM WAV 音频"

    if channels not in {1, 2}:
        return "录音必须是单声道或双声道 WAV 音频"
    if sample_width != 2:
        return "录音必须使用 16-bit PCM 编码"
    if sample_rate < 16_000:
        return "录音采样率不能低于 16 kHz"
    duration_seconds = frame_count / sample_rate
    if duration_seconds < 5 or duration_seconds > 60:
        return "录音时长必须在 5–60 秒之间"
    return None


def _settings_for(websocket: WebSocket) -> Settings:
    settings = getattr(websocket.app.state, "settings", None)
    if settings is None:
        raise RuntimeError("application settings are not initialized")
    return settings


def _api_key(settings: Settings) -> str:
    configured = getattr(settings, "tts_api_key", None)
    if configured is None:
        raise RuntimeError("FISH_MEMORY_TTS_API_KEY is not configured")
    return configured.get_secret_value()


def _websocket_url(settings: Settings) -> str:
    return getattr(
        settings,
        "tts_websocket_url",
        DEFAULT_DASHSCOPE_WEBSOCKET_URL,
    )


def _connection_key(request: TtsStartRequest) -> TtsConnectionKey:
    return (
        request.model,
        request.voice,
        request.audio_format,
        request.sample_rate,
    )


def _validate_speech_tag_request(request: TtsStartRequest, provider: str) -> None:
    if request.speech_tags and not supports_speech_tags(provider, request.model):
        raise ValueError(
            "speech tags are not supported by the selected TTS provider/model"
        )


def _parse_client_task_message(
    message: dict[str, Any],
    last_request: TtsStartRequest | None,
) -> tuple[TtsStartRequest | None, str | None, bool, bool]:
    """Parse a client command and tolerate controls queued behind a terminal event."""
    message_type = message.get("type")
    if message_type == "start":
        request = TtsStartRequest.model_validate(message)
        initial_text = message.get("text")
        if initial_text is not None:
            initial_text = TtsRequest.model_validate(
                {**message, "text": initial_text}
            ).text
        return request, initial_text, True, False

    if last_request is not None and message_type == "append":
        append = TtsAppendRequest.model_validate(message)
        return last_request, append.text, False, False

    if last_request is not None and message_type in {"flush", "finish", "stop"}:
        return None, None, True, True

    raise ValueError(f"message must be a start message, got {message_type!r}")


async def _send_error(
    websocket: WebSocket, message: str, error: BaseException | None = None
) -> None:
    body: dict[str, object] = {"type": "error", "message": message}
    if error is not None:
        body["failure"] = failures.classified(
            failures.noted(error, "voice"), "voice"
        ).code
    with suppress(Exception):
        await websocket.send_json(body)
    with suppress(Exception):
        await websocket.close(code=1011)


async def _wait_for_control(
    websocket: WebSocket,
    session: DashScopeTtsSession,
    stop_event: asyncio.Event,
    timing: dict[str, float | None],
    facial: SpeechFacialPlan | None = None,
) -> str:
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=TTS_IDLE_ROTATION_INTERVAL_SECONDS,
                )
            except TimeoutError:
                await session.send(build_finish_task(session.task_id))
                return "rotate"
            message_type = message.get("type")
            if message_type == "append":
                append = TtsAppendRequest.model_validate(message)
                if timing["text_sent_at"] is None:
                    timing["text_sent_at"] = time.perf_counter()
                if facial is not None:
                    facial.append(append.text)
                await session.send(build_continue_task(session.task_id, append.text))
                continue
            if message_type == "finish":
                await session.send(build_finish_task(session.task_id))
                return "finish"
            if message_type == "flush":
                # The application has a complete semantic segment, even when
                # its text has no punctuation. Commit it on the same connection
                # and wait for the continuation without ending client playback.
                await session.send(build_finish_task(session.task_id))
                return "rotate"
            if message_type == "stop":
                stop_event.set()
                if facial is not None:
                    facial.cancel_pending()
                await session.send(build_cancel_task(session.task_id))
                return "stop"
            raise ValueError("message must be append, flush, finish, or stop")
    except (WebSocketDisconnect, RuntimeError, ValidationError, ValueError):
        stop_event.set()
        if facial is not None:
            facial.cancel_pending()
        with suppress(Exception):
            await session.send(build_cancel_task(session.task_id))
        return "stop"


async def _wait_for_next_text(
    websocket: WebSocket,
    stop_event: asyncio.Event,
) -> tuple[str, str | None]:
    """Wait without an active provider task after an idle task is rotated."""
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "append":
                append = TtsAppendRequest.model_validate(message)
                return "append", append.text
            if message_type == "flush":
                continue
            if message_type == "finish":
                return "finish", None
            if message_type == "stop":
                stop_event.set()
                return "stop", None
            raise ValueError("message must be append, flush, finish, or stop")
    except (WebSocketDisconnect, RuntimeError, ValidationError, ValueError):
        stop_event.set()
        return "stop", None


def _browser_event(event: TtsServerEvent) -> dict[str, Any] | None:
    if event.kind == "sentence_begin":
        return {
            "type": "sentence_begin",
            "sentenceIndex": event.sentence_index,
            "text": event.text or "",
        }
    if event.kind == "sentence_end":
        return {
            "type": "sentence_end",
            "sentenceIndex": event.sentence_index,
            "text": event.text or "",
        }
    return None


async def _forward_provider_events(
    websocket: WebSocket,
    session: DashScopeTtsSession,
    *,
    started_at: float,
    timing: dict[str, float | None],
    stop_event: asyncio.Event,
    facial: SpeechFacialPlan | None = None,
) -> float | None:
    websocket = _serialized(websocket)
    if facial is not None:
        facial.set_sender(websocket.send_json)
    first_audio_at: float | None = None
    pending_audio: TtsServerEvent | None = None
    active_sentence: TtsServerEvent | None = None
    while True:
        raw_event = await session.receive()
        if isinstance(raw_event, bytes):
            if pending_audio is None:
                continue
            now = time.perf_counter()
            is_first_audio = first_audio_at is None
            if first_audio_at is None:
                first_audio_at = now
            text_sent_at = timing["text_sent_at"] or started_at
            audio_index, audio_text = (
                pending_audio.sentence_index,
                pending_audio.text or "",
            )
            facial_wait_ms = 0.0
            if facial is not None:
                if active_sentence is not None and (
                    audio_index is None or audio_index == active_sentence.sentence_index
                ):
                    audio_index = active_sentence.sentence_index
                    audio_text = active_sentence.text or audio_text
                cue = await facial.before_audio(audio_index, audio_text)
                if stop_event.is_set():
                    return first_audio_at
                if cue is not None:
                    facial_wait_ms = cue["waitMs"]
                    await websocket.send_json(cue)
            await websocket.send_packet(
                {
                    "type": "audio",
                    "sentenceIndex": audio_index,
                    "text": audio_text,
                    "firstPacketMs": round((now - text_sent_at) * 1000, 1)
                    if is_first_audio
                    else None,
                    "firstPacketFromStartMs": round((now - started_at) * 1000, 1)
                    if is_first_audio
                    else None,
                    "facialWaitMs": facial_wait_ms if is_first_audio else None,
                },
                raw_event,
            )
            pending_audio = None
            continue

        if not isinstance(raw_event, str):
            continue
        try:
            decoded = json.loads(raw_event)
        except ValueError:
            continue
        if not isinstance(decoded, dict):
            continue
        event = parse_server_event(decoded)
        if event.kind == "audio":
            pending_audio = event
            continue
        if event.kind == "failed":
            raise event.failure()
        if event.kind == "sentence_begin" and facial is not None:
            active_sentence = event
            facial.begin(event.sentence_index, event.text or "")
        if event.kind == "sentence_end" and facial is not None:
            facial.end(event.sentence_index)
            active_sentence = None
        browser_event = _browser_event(event)
        if browser_event is not None:
            await websocket.send_json(browser_event)
        if event.kind == "finished":
            return first_audio_at


async def _send_finished_event(
    websocket: WebSocket,
    *,
    started_at: float,
    timing: dict[str, float | None],
    first_audio_at: float | None,
    stopped: bool,
) -> None:
    finished_at = time.perf_counter()
    text_sent_at = timing["text_sent_at"] or started_at
    await websocket.send_json(
        {
            "type": "finished",
            "stopped": stopped,
            "firstAudioMs": round((first_audio_at - text_sent_at) * 1000, 1)
            if first_audio_at is not None
            else None,
            "totalMs": round((finished_at - started_at) * 1000, 1),
        }
    )


async def _try_send_stopped_event(
    websocket: WebSocket,
    *,
    started_at: float,
    timing: dict[str, float | None],
    first_audio_at: float | None,
) -> None:
    """Best-effort terminal event after cancellation or client disconnect."""
    with suppress(WebSocketDisconnect, RuntimeError):
        await _send_finished_event(
            websocket,
            started_at=started_at,
            timing=timing,
            first_audio_at=first_audio_at,
            stopped=True,
        )


async def _wait_for_task_started(session: DashScopeTtsSession) -> None:
    while True:
        raw_event = await session.receive()
        if isinstance(raw_event, bytes):
            continue
        if not isinstance(raw_event, str):
            continue
        decoded = json.loads(raw_event)
        if not isinstance(decoded, dict):
            continue
        event = parse_server_event(decoded)
        if event.kind == "failed":
            raise event.failure()
        if event.kind == "started":
            return


async def _ensure_provider_session(
    settings: Settings,
    request: TtsStartRequest,
    session: DashScopeTtsSession | None,
    current_key: TtsConnectionKey | None,
) -> tuple[DashScopeTtsSession, TtsConnectionKey, bool, float | None]:
    requested_key = _connection_key(request)
    if session is not None and current_key == requested_key and session.is_connected:
        return session, requested_key, True, None

    if session is not None:
        await session.close()

    next_session = DashScopeTtsSession(
        websocket_url=_websocket_url(settings),
        api_key=_api_key(settings),
    )
    connect_started_at = time.perf_counter()
    await next_session.connect()
    connection_ms = (time.perf_counter() - connect_started_at) * 1000
    return next_session, requested_key, False, connection_ms


async def _run_tts_task(
    websocket: WebSocket,
    session: DashScopeTtsSession,
    request: TtsStartRequest,
    *,
    started_at: float,
    connection_reused: bool,
    connection_ms: float | None,
    initial_text: str | None = None,
    initial_display_text: str | None = None,
    initial_text_is_final: bool = True,
    facial: SpeechFacialPlan | None = None,
) -> bool:
    _validate_speech_tag_request(request, "dashscope")
    if request.speech_tags and supports_speech_tags("dashscope", request.model):
        return await run_tagged_tts(
            _serialized(websocket),
            session,
            request,
            started_at=started_at,
            initial_text=initial_text,
            initial_display_text=initial_display_text,
            initial_text_is_final=initial_text_is_final,
            connection_reused=connection_reused,
            connection_ms=connection_ms,
            facial=facial,
            idle_timeout=TTS_IDLE_ROTATION_INTERVAL_SECONDS,
        )
    return await _run_tts_task_prepared(
        _serialized(websocket),
        session,
        request,
        started_at=started_at,
        initial_text=initial_text,
        initial_text_is_final=initial_text_is_final,
        connection_reused=connection_reused,
        connection_ms=connection_ms,
        facial=facial,
    )


async def _run_tts_task_prepared(
    websocket: WebSocket,
    session: DashScopeTtsSession,
    request: TtsStartRequest,
    *,
    started_at: float,
    connection_reused: bool,
    connection_ms: float | None,
    initial_text: str | None = None,
    initial_text_is_final: bool = True,
    facial: SpeechFacialPlan | None = None,
) -> bool:
    stop_event = asyncio.Event()
    timing: dict[str, float | None] = {"text_sent_at": None}
    pending_text = initial_text
    is_first_task = True
    while True:
        task_id = session.begin_task()
        await session.send(build_run_task(request, task_id=task_id))
        await _wait_for_task_started(session)

        if is_first_task:
            ready_at = time.perf_counter()
            await websocket.send_json(
                {
                    "type": "ready",
                    "connectMs": round((ready_at - started_at) * 1000, 1),
                    "connectionMs": (
                        round(connection_ms, 1) if connection_ms is not None else None
                    ),
                    "connectionReused": connection_reused,
                    "provider": "dashscope",
                    "model": request.model,
                    "voice": request.voice,
                    "sampleRate": request.sample_rate,
                    **(
                        {"facialStreamId": facial.stream_id}
                        if facial is not None
                        else {}
                    ),
                }
            )

        if pending_text is not None:
            if facial is not None:
                facial.append(pending_text)
            timing["text_sent_at"] = timing["text_sent_at"] or time.perf_counter()
            await session.send(build_continue_task(task_id, pending_text))
            if is_first_task and initial_text is not None and initial_text_is_final:
                await session.send(build_finish_task(task_id))
            pending_text = None

        control_task = asyncio.create_task(
            _wait_for_control(websocket, session, stop_event, timing, facial)
        )
        provider_task: asyncio.Task[float | None] = asyncio.create_task(
            _forward_provider_events(
                websocket,
                session,
                started_at=started_at,
                timing=timing,
                stop_event=stop_event,
                facial=facial,
            )
        )
        control_result: str | None = None
        try:
            done, _ = await asyncio.wait(
                {control_task, provider_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if control_task in done:
                control_result = await control_task
            if stop_event.is_set():
                try:
                    first_audio_at = await asyncio.wait_for(
                        provider_task,
                        timeout=TTS_CANCEL_TIMEOUT_SECONDS,
                    )
                except Exception as error:
                    logger.info(
                        "TTS provider connection discarded after cancellation: %s",
                        error,
                    )
                    first_audio_at = None
                await _try_send_stopped_event(
                    websocket,
                    started_at=started_at,
                    timing=timing,
                    first_audio_at=first_audio_at,
                )
                return False
            else:
                first_audio_at = await provider_task
        except Exception as error:
            if not stop_event.is_set():
                raise
            logger.info(
                "TTS provider connection discarded after cancellation: %s",
                error,
            )
            await _try_send_stopped_event(
                websocket,
                started_at=started_at,
                timing=timing,
                first_audio_at=None,
            )
            return False
        finally:
            for task in (control_task, provider_task):
                if task.done():
                    continue
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        if control_result == "rotate":
            if facial is not None:
                facial.rotate()
            await websocket.send_json({"type": "segment_finished"})
            next_result, pending_text = await _wait_for_next_text(
                websocket,
                stop_event,
            )
            if next_result == "append":
                is_first_task = False
                continue
            if next_result == "stop" and facial is not None:
                facial.cancel_pending()
            await _send_finished_event(
                websocket,
                started_at=started_at,
                timing=timing,
                first_audio_at=first_audio_at,
                stopped=next_result == "stop",
            )
            return next_result != "stop"

        await _send_finished_event(
            websocket,
            started_at=started_at,
            timing=timing,
            first_audio_at=first_audio_at,
            stopped=False,
        )
        return True


def _local_service_for(websocket: WebSocket, settings: Settings) -> LocalTtsService:
    service = getattr(websocket.app.state, "local_tts_service", None)
    if service is None:
        service = LocalTtsService(LocalTtsModelConfig.from_settings(settings))
        websocket.app.state.local_tts_service = service
    return service


async def _run_local_tts_task(
    websocket: WebSocket,
    service: LocalTtsService,
    request: TtsStartRequest,
    *,
    started_at: float,
    initial_text: str | None = None,
    initial_text_is_final: bool = True,
    facial: SpeechFacialPlan | None = None,
) -> None:
    return await _run_local_tts_task_prepared(
        _serialized(websocket),
        service,
        request,
        started_at=started_at,
        initial_text=initial_text,
        initial_text_is_final=initial_text_is_final,
        facial=facial,
    )


async def _run_local_tts_task_prepared(
    websocket: WebSocket,
    service: LocalTtsService,
    request: TtsStartRequest,
    *,
    started_at: float,
    initial_text: str | None = None,
    initial_text_is_final: bool = True,
    facial: SpeechFacialPlan | None = None,
) -> None:
    websocket = _serialized(websocket)
    if facial is not None:
        facial.set_sender(websocket.send_json)
    was_ready = bool(getattr(service, "is_ready", False))
    await service.prepare()
    ready_at = time.perf_counter()
    connection_ms = None
    if not was_ready:
        connection_ms = round((ready_at - started_at) * 1000, 1)
    await websocket.send_json(
        {
            "type": "ready",
            "connectMs": round((ready_at - started_at) * 1000, 1),
            "connectionMs": connection_ms,
            "connectionReused": was_ready,
            "provider": "local",
            "model": service.model_name,
            "voice": "default",
            "sampleRate": request.sample_rate,
            **({"facialStreamId": facial.stream_id} if facial is not None else {}),
        }
    )

    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_event = threading.Event()
    timing: dict[str, float | None] = {"text_sent_at": None}
    first_audio_at: float | None = None

    async def synthesize() -> None:
        nonlocal first_audio_at
        sentence_index = 0
        while True:
            text = await text_queue.get()
            if text is None or stop_event.is_set():
                return
            await websocket.send_json(
                {
                    "type": "sentence_begin",
                    "sentenceIndex": sentence_index,
                    "text": text,
                }
            )
            face_sent = False
            if facial is not None:
                facial.begin(sentence_index, text)
            async for pcm in service.stream(
                text,
                sample_rate=request.sample_rate,
                stop_event=stop_event,
            ):
                if stop_event.is_set():
                    return
                now = time.perf_counter()
                is_first_audio = first_audio_at is None
                if is_first_audio:
                    first_audio_at = now
                text_sent_at = timing["text_sent_at"] or started_at
                facial_wait_ms = 0.0
                if facial is not None and not face_sent:
                    cue = await facial.before_audio(sentence_index, text)
                    face_sent = True
                    if stop_event.is_set():
                        return
                    if cue is not None:
                        facial_wait_ms = cue["waitMs"]
                        await websocket.send_json(cue)
                await websocket.send_packet(
                    {
                        "type": "audio",
                        "sentenceIndex": sentence_index,
                        "text": text,
                        "firstPacketMs": round((now - text_sent_at) * 1000, 1)
                        if is_first_audio
                        else None,
                        "firstPacketFromStartMs": round((now - started_at) * 1000, 1)
                        if is_first_audio
                        else None,
                        "facialWaitMs": facial_wait_ms if is_first_audio else None,
                    },
                    pcm,
                )
            if stop_event.is_set():
                return
            await websocket.send_json(
                {
                    "type": "sentence_end",
                    "sentenceIndex": sentence_index,
                    "text": text,
                }
            )
            sentence_index += 1

    worker = asyncio.create_task(synthesize())
    stopped = False
    try:
        if initial_text is not None:
            timing["text_sent_at"] = time.perf_counter()
            if facial is not None:
                facial.append(initial_text)
            await text_queue.put(initial_text)
            if initial_text_is_final:
                await text_queue.put(None)
        if not (initial_text is not None and initial_text_is_final):
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "append":
                    append = TtsAppendRequest.model_validate(message)
                    timing["text_sent_at"] = (
                        timing["text_sent_at"] or time.perf_counter()
                    )
                    if facial is not None:
                        facial.append(append.text)
                    await text_queue.put(append.text)
                    continue
                if message_type == "flush":
                    # Local synthesis already treats each append as a segment.
                    continue
                if message_type == "finish":
                    await text_queue.put(None)
                    break
                if message_type == "stop":
                    stopped = True
                    stop_event.set()
                    if facial is not None:
                        facial.cancel_pending()
                    while not text_queue.empty():
                        text_queue.get_nowait()
                    await text_queue.put(None)
                    break
                raise ValueError("message must be append, flush, finish, or stop")
        await worker
    except (WebSocketDisconnect, RuntimeError, ValidationError, ValueError):
        stopped = True
        stop_event.set()
        if not worker.done():
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        raise

    await _send_finished_event(
        websocket,
        started_at=started_at,
        timing=timing,
        first_audio_at=first_audio_at,
        stopped=stopped,
    )


@tts_router.websocket("/stream")
async def tts_stream(websocket: WebSocket) -> None:
    websocket = _serialized(websocket)
    await websocket.accept()
    session: DashScopeTtsSession | None = None
    session_key: TtsConnectionKey | None = None
    session_credentials: tuple[str, str] | None = None
    last_request: TtsStartRequest | None = None
    facial: SpeechFacialPlan | None = None
    try:
        while True:
            message = await websocket.receive_json()
            # A running task owns its old settings until completion. The next
            # task on this same client socket picks up saved provider/keys.
            settings = _settings_for(websocket).model_copy()
            if not isinstance(message, dict):
                raise ValueError("TTS message must be a JSON object")
            request, initial_text, initial_text_is_final, stale = (
                _parse_client_task_message(message, last_request)
            )
            if stale:
                if message.get("type") == "stop" and facial is not None:
                    await facial.close()
                    facial = None
                # The corresponding terminal event was already sent to the
                # client. This control message was queued behind that event.
                logger.debug("TTS ignored stale terminal control message")
                continue
            if request is None:
                raise RuntimeError("TTS task request is missing")
            if initial_text is not None and not initial_text_is_final:
                logger.debug(
                    "TTS accepted append after provider task finished as a new task"
                )
            if initial_text_is_final:
                last_request = request
            if initial_text_is_final or facial is None:
                if facial is not None:
                    await facial.close()
                facial = create_speech_facial_plan(
                    websocket,
                    request.facial,
                    wait_for_ready=settings.speech_facial_wait_for_ready,
                )
            else:
                # An append queued behind provider completion is a continuation
                # of the same native utterance. Preserve cumulative offsets,
                # policy snapshot, and pending faces for PCM still playing.
                facial.rotate()
            # The voice itself says which provider speaks it, not the default.
            provider = model_provider(request.model, settings)
            _validate_speech_tag_request(request, provider)
            if provider == "local":
                if session is not None:
                    await session.close()
                    session = None
                    session_key = None
                local_service = _local_service_for(websocket, settings)
                started_at = time.perf_counter()
                await websocket.send_json(
                    {
                        "type": "connecting",
                        "connectionReused": local_service.is_ready,
                        "provider": "local",
                    }
                )
                await _run_local_tts_task(
                    websocket,
                    local_service,
                    request,
                    started_at=started_at,
                    initial_text=initial_text,
                    initial_text_is_final=initial_text_is_final,
                    facial=facial,
                )
                failures.succeeded("voice")
                continue
            model = find_tts_model(request.model)
            if model is None:
                raise ValueError(f"unsupported TTS model: {request.model}")
            requested_voice = FishVoiceConfig(
                model=request.model,
                voice=request.voice,
                kind=(
                    VoiceKind.PRESET
                    if any(voice.id == request.voice for voice in model.voices)
                    else VoiceKind.CUSTOM
                ),
            )
            if not is_supported_voice(requested_voice, get_tts_catalog()):
                raise ValueError(f"unsupported voice: {request.voice}")

            requested_key = _connection_key(request)
            credentials = (_websocket_url(settings), _api_key(settings))
            if session is not None and session_credentials != credentials:
                await session.close()
                session = None
                session_key = None
            session_credentials = credentials
            connection_reused = (
                session is not None
                and session_key == requested_key
                and session.is_connected
            )
            started_at = time.perf_counter()
            await websocket.send_json(
                {
                    "type": "connecting",
                    "connectionReused": connection_reused,
                    "provider": "dashscope",
                }
            )
            (
                session,
                session_key,
                connection_reused,
                connection_ms,
            ) = await _ensure_provider_session(
                settings,
                request,
                session,
                session_key,
            )
            try:
                provider_reusable = await _run_tts_task(
                    websocket,
                    session,
                    request,
                    started_at=started_at,
                    connection_reused=connection_reused,
                    connection_ms=connection_ms,
                    initial_text=initial_text,
                    initial_display_text=message.get("display_text"),
                    initial_text_is_final=initial_text_is_final,
                    facial=facial,
                )
                failures.succeeded("voice")
                if not provider_reusable:
                    await session.close()
                    session = None
                    session_key = None
            except Exception:
                if session is not None:
                    await session.close()
                session = None
                session_key = None
                raise
    except WebSocketDisconnect:
        pass
    except RuntimeError as error:
        if not _is_disconnected_websocket_error(error):
            logger.exception("TTS WebSocket/provider failed")
            await _send_error(websocket, str(error), error)
    except (ValidationError, ValueError) as error:
        logger.warning("TTS WebSocket request rejected: %s", error)
        await _send_error(websocket, str(error), error)
    except Exception as error:  # pragma: no cover - network/provider boundary
        logger.exception("TTS WebSocket unexpected failure")
        await _send_error(websocket, f"TTS stream failed: {error}", error)
    finally:
        if facial is not None:
            await facial.close()
        if session is not None:
            await session.close()
        with suppress(Exception):
            await websocket.close()


def _is_disconnected_websocket_error(error: RuntimeError) -> bool:
    """Recognize Starlette's post-disconnect receive guard.

    Starlette raises RuntimeError, rather than WebSocketDisconnect, when a
    receive is attempted after it has already delivered the disconnect frame.
    This is a normal client close and must not be reported as a provider fault.
    """

    return str(error).startswith("WebSocket is not connected.")
