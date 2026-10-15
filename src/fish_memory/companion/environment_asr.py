"""Serve local media recognition when interactive microphone voice is off."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from fish_memory.companion.asr import (
    AsrModelConfig,
    AsrModelDownloader,
    AsrUpdate,
    StreamingAsr,
)
from fish_memory.companion.shared_asr import (
    SharedAsrServer,
    configured_shared_asr_socket,
)
from fish_memory.runtime_platform import shutdown_handlers

logger = logging.getLogger(__name__)


class EnvironmentAsrRecognizer:
    """Lazy single model, called only by the shared server's utility worker."""

    def __init__(self, settings: object) -> None:
        self._config = replace(AsrModelConfig.from_settings(settings), num_threads=1)
        self._decoder: StreamingAsr | None = None

    @property
    def model(self) -> str:
        return self._config.model

    def buffered_pcm_bytes(self) -> int:
        return 0

    def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate | None:
        if self._decoder is None:
            AsrModelDownloader(self._config).ensure()
            self._decoder = StreamingAsr(self._config)
        return self._decoder.try_recognize_snapshot_update(pcm)


async def run_environment_asr(
    settings: object, *, stop_event: asyncio.Event | None = None
) -> None:
    path = configured_shared_asr_socket()
    if path is None:
        raise RuntimeError("environment ASR requires a shared socket")
    stop = stop_event if stop_event is not None else asyncio.Event()
    server = SharedAsrServer(path)
    server.bind(EnvironmentAsrRecognizer(settings))
    with shutdown_handlers(
        stop.set, signals=stop_event is None, parent=stop_event is None
    ):
        try:
            await server.start()
            logger.info("影音转录已就绪，麦克风交流保持关闭")
            await stop.wait()
        finally:
            # Drain inference before handing the endpoint to Companion.
            await server.close()
