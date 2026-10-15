import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ProcessorProtocol(Protocol):
    async def process_once(self) -> bool: ...


class BackgroundTaskSupervisor:
    def __init__(
        self,
        processor: ProcessorProtocol,
        *,
        name: str,
        poll_interval_seconds: float,
    ) -> None:
        self._processor = processor
        self._name = name
        self._poll_interval_seconds = poll_interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(),
            name=f"fish-memory-{self._name}",
        )

    def configure_poll_interval(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("poll interval must be positive")
        self._poll_interval_seconds = seconds

    async def close(self) -> None:
        if self._task is None:
            return
        task = self._task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                worked = await self._processor.process_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("background %s failed", self._name)
                worked = False
            if not worked:
                await asyncio.sleep(self._poll_interval_seconds)
            else:
                await asyncio.sleep(0)
