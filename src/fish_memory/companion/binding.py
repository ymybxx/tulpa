import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class AsyncResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class AsyncHttpClient(Protocol):
    async def get(self, url: str) -> AsyncResponse: ...

    async def aclose(self) -> None: ...


class BindingSource(Protocol):
    async def get_current(self) -> UUID | None: ...


class AssistantBindingClient:
    """Reads the persisted assistant Fish binding without creating Fish records."""

    def __init__(
        self,
        api_url: str,
        http_client: AsyncHttpClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._http_client = http_client or httpx.AsyncClient()
        if api_key:
            headers = getattr(self._http_client, "headers", None)
            if headers is None:
                raise TypeError("authenticated HTTP client must expose headers")
            headers["Authorization"] = f"Bearer {api_key}"
        self._owns_client = http_client is None

    async def get_current(self) -> UUID | None:
        response = await self._http_client.get(f"{self._api_url}/v1/fishes/assistant")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("assistant binding response must be an object")
        fish_id = payload.get("fishId")
        if fish_id is None:
            return None
        if not isinstance(fish_id, str):
            raise ValueError("assistant binding fishId must be a UUID or null")
        return UUID(fish_id)

    async def close(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()


class AssistantBindingWatcher:
    """Polls the binding and reports only successful value transitions."""

    def __init__(
        self,
        client: BindingSource,
        *,
        poll_interval_seconds: float = 1.0,
        bound_poll_interval_seconds: float | None = None,
    ) -> None:
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if bound_poll_interval_seconds is not None and bound_poll_interval_seconds < 0:
            raise ValueError("bound_poll_interval_seconds must be non-negative")
        self._client = client
        self._poll_interval_seconds = poll_interval_seconds
        self._bound_poll_interval_seconds = (
            poll_interval_seconds
            if bound_poll_interval_seconds is None
            else bound_poll_interval_seconds
        )
        self._closed = asyncio.Event()
        self._last_successful: UUID | None = None
        self._last_notified: UUID | None | object = _UNSET

    async def poll_once(self) -> UUID | None:
        current = await self._client.get_current()
        self._last_successful = current
        return current

    async def run(
        self,
        on_change: Callable[[UUID | None], Awaitable[None]],
    ) -> None:
        while not self._closed.is_set():
            try:
                current = await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unable to read assistant Fish binding")
            else:
                if self._last_notified is _UNSET or current != self._last_notified:
                    try:
                        await on_change(current)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Do not commit a binding that the runner failed to
                        # apply. Polling the same value again retries the full
                        # resource transition instead of leaving audio bound to
                        # the previous Fish forever.
                        logger.exception(
                            "Unable to apply assistant Fish binding %s",
                            current,
                        )
                    else:
                        self._last_notified = current

            if self._closed.is_set():
                break
            delay = (
                self._bound_poll_interval_seconds
                if self._last_successful is not None
                else self._poll_interval_seconds
            )
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=delay)
            except TimeoutError:
                pass

    def close(self) -> None:
        self._closed.set()


_UNSET = object()
