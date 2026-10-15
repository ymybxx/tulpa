"""Explicit desktop enrollment capture, with the production input backend."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from fish_memory.companion.owner_voice import (
    MAX_ENROLL_SECONDS,
    MIN_ENROLL_SECONDS,
    SAMPLE_RATE,
    OwnerVoiceError,
)
from fish_memory.config import Settings


async def capture_owner_voice(
    settings: Settings,
    *,
    duration_seconds: int,
    on_recording: Callable[[], None],
) -> bytes:
    """Capture in memory only; never open KWS/ASR or submit a conversation."""
    from fish_memory.companion.audio_processing import WebRtcAec
    from fish_memory.companion.runner import _default_audio_backend

    if not MIN_ENROLL_SECONDS <= duration_seconds <= MAX_ENROLL_SECONDS:
        raise OwnerVoiceError("主人声纹录音时长必须在 6 至 30 秒之间")
    audio_input, _output = _default_audio_backend(settings)
    aec = (
        None
        if getattr(audio_input, "capture_is_preprocessed", False)
        else WebRtcAec(
            stream_delay_ms=settings.companion_aec_delay_ms,
            delay_mode=settings.companion_aec_delay_mode,
        )
    )
    stopped = asyncio.Event()
    captured = bytearray()
    target_bytes = SAMPLE_RATE * 2 * duration_seconds
    started = False

    async def on_frame(pcm: bytes, _render: bytes) -> None:
        nonlocal started
        if not isinstance(pcm, bytes) or len(pcm) != 320:
            raise OwnerVoiceError("采集到的录音帧格式无效")
        if not started:
            if aec is not None and settings.companion_aec_delay_mode == "auto":
                device_delay = audio_input.stream_delay_ms
                if device_delay is not None:
                    aec.set_stream_delay(device_delay)
            started = True
            on_recording()
        if aec is not None:

            def clean() -> bytes:
                aec.process_reverse_stream(_render)
                return bytes(aec.process_stream(pcm))

            pcm = await asyncio.to_thread(clean)
            if len(pcm) != 320:
                raise OwnerVoiceError("预处理录音帧格式无效")
        remaining = target_bytes - len(captured)
        captured.extend(pcm[:remaining])
        if len(captured) >= target_bytes:
            stopped.set()

    try:
        # Both backends have their own first-frame/stall watchdog. This also
        # bounds an input backend that keeps running but delivers too few frames.
        async with asyncio.timeout(duration_seconds + 15):
            await audio_input.run(on_frame, stopped)
        if len(captured) != target_bytes:
            raise OwnerVoiceError("录音意外结束，请重试")
        if audio_input.dropped_frames:
            raise OwnerVoiceError("录音期间发生丢帧，请重试")
        return bytes(captured)
    finally:
        stopped.set()
        # close(), not playback stop(): it terminates/reaps the VPIO helper.
        await audio_input.close()
        captured.clear()
