import asyncio
import json
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from fish_memory.kws.detector import AudioFormatError, KwsUnavailableError
from fish_memory.kws.models import KwsStartRequest

kws_router = APIRouter(prefix="/v1/kws", tags=["public", "kws"])


async def _send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await websocket.send_json({"type": "error", "message": message})
    with suppress(Exception):
        await websocket.close(code=1011)


@kws_router.get("/status")
async def kws_status(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "kws_service", None)
    if service is None:
        return {
            "available": False,
            "downloading": False,
            "wakeWord": "小鱼",
            "sampleRate": 16_000,
            "channels": 1,
            "chunkSize": 8,
            "modelDir": None,
            "error": "KWS service is not initialized",
            "downloadedBytes": 0,
            "totalBytes": None,
            "downloadPercent": None,
        }
    return service.status()


@kws_router.websocket("/stream")
async def kws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    connection: Any = None
    try:
        first_message = await websocket.receive()
        if first_message.get("bytes") is not None:
            await _send_error(websocket, "first message must be a JSON start message")
            return
        raw_text = first_message.get("text")
        if raw_text is None:
            await _send_error(websocket, "missing start message")
            return
        try:
            start = KwsStartRequest.model_validate(json.loads(raw_text))
            if start.sample_rate != 16_000:
                raise ValueError("sampleRate must be 16000")
            if start.channels != 1:
                raise ValueError("channels must be 1")
            if start.format != "s16le":
                raise ValueError("format must be s16le")
            service = websocket.app.state.kws_service
            connection = await service.open_stream(start.wake_word)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            await _send_error(websocket, str(error))
            return
        except KwsUnavailableError as error:
            await _send_error(websocket, str(error))
            return

        await websocket.send_json(
            {
                "type": "ready",
                "wakeWord": start.wake_word,
                "sampleRate": 16_000,
                "channels": 1,
                "format": "s16le",
            }
        )
        audio_samples = 0
        processing_ms = 0.0
        last_stats_at = time.perf_counter()
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                pcm_bytes = message["bytes"]
                if len(pcm_bytes) % 2:
                    raise AudioFormatError("PCM must contain complete 16-bit samples")
                detections = await asyncio.to_thread(
                    connection.accept_pcm,
                    pcm_bytes,
                )
                audio_samples += len(pcm_bytes) // 2
                processing_ms += sum(item.processing_ms for item in detections)
                now = time.perf_counter()
                if now - last_stats_at >= 0.25 or detections:
                    await websocket.send_json(
                        {
                            "type": "stats",
                            "audioMs": round(audio_samples * 1000 / 16_000, 1),
                            "processingMs": round(processing_ms, 3),
                        }
                    )
                    last_stats_at = now
                for detection in detections:
                    await websocket.send_json(
                        {
                            "type": "detected",
                            "keyword": detection.keyword,
                            "audioOffsetMs": detection.audio_offset_ms,
                            "processingMs": detection.processing_ms,
                        }
                    )
                continue

            raw_control = message.get("text")
            if raw_control is None:
                continue
            control = json.loads(raw_control)
            if control.get("type") == "reset":
                await asyncio.to_thread(connection.reset)
                await websocket.send_json({"type": "reset"})
                continue
            if control.get("type") == "stop":
                return
            await _send_error(websocket, "control message must be reset or stop")
            return
    except WebSocketDisconnect:
        return
    except (AudioFormatError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        await _send_error(websocket, str(error))
    finally:
        if connection is not None:
            with suppress(Exception):
                await asyncio.to_thread(connection.close)
