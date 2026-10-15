"""Lease custom-voice clients across credential changes without canceling uploads."""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fish_memory.config import Settings
from fish_memory.tts.custom_voices import CustomVoice, DashScopeVoiceClient


@dataclass(eq=False)
class _VoiceClient:
    client: DashScopeVoiceClient
    users: int = 0
    retired: bool = False


class CustomVoiceRuntime:
    def __init__(
        self, settings: Settings, *, factory: Callable[..., DashScopeVoiceClient]
    ) -> None:
        self.settings = settings
        self.factory = factory
        self._current: _VoiceClient | None = None
        self._cleanup: set[asyncio.Task[None]] = set()
        self._closed = False

    def invalidate(self) -> None:
        current, self._current = self._current, None
        if current is not None:
            current.retired = True
            self._retire_idle(current)

    @contextmanager
    def _borrow(self) -> Iterator[DashScopeVoiceClient]:
        if self._closed:
            raise RuntimeError("custom voice service is closed")
        key = self.settings.tts_api_key
        if key is None:
            raise RuntimeError("FISH_MEMORY_TTS_API_KEY is not configured")
        if self._current is None:
            self._current = _VoiceClient(
                self.factory(
                    api_key=key.get_secret_value(),
                    customization_url=self.settings.tts_customization_url,
                    upload_url=self.settings.tts_upload_url,
                )
            )
        current = self._current
        current.users += 1
        try:
            yield current.client
        finally:
            current.users -= 1
            self._retire_idle(current)

    def _retire_idle(self, resource: _VoiceClient) -> None:
        if resource.retired and not resource.users:
            task = asyncio.create_task(resource.client.close())
            self._cleanup.add(task)
            task.add_done_callback(self._closed_client)

    def _closed_client(self, task: asyncio.Task[None]) -> None:
        self._cleanup.discard(task)
        if not task.cancelled():
            task.exception()

    async def list_voices(self) -> list[CustomVoice]:
        with self._borrow() as client:
            return await client.list_voices()

    async def create_voice(self, audio: bytes, *, target_model: str) -> CustomVoice:
        with self._borrow() as client:
            return await client.create_voice(audio, target_model=target_model)

    async def delete_voice(self, voice_id: str) -> None:
        with self._borrow() as client:
            await client.delete_voice(voice_id)

    async def close(self) -> None:
        self._closed = True
        self.invalidate()
        if self._cleanup:
            await asyncio.gather(*tuple(self._cleanup), return_exceptions=True)
