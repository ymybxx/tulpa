"""Replace desktop model routes while keeping in-flight requests alive."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fish_memory import failures
from fish_memory.llm.configuration import DesktopModelRoutes, LanguageModelEndpoint
from fish_memory.llm.logs import call_scope
from fish_memory.llm.streaming import closing_iterator

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class _Client:
    value: Any
    references: int = 0


@dataclass(eq=False, repr=False)
class _Generation:
    routes: DesktopModelRoutes
    chat: _Client
    reasoning: _Client
    environment: _Client
    references: int = 1

    def clients(self) -> set[_Client]:
        return {self.chat, self.reasoning, self.environment}


class ModelRoute:
    """Stable dependency with one generation lease per provider operation."""

    def __init__(self, owner: ModelRuntime, name: str) -> None:
        self._owner = owner
        self._name = name

    async def close(self) -> None:
        # The application owns all routes together, including shared clients.
        # A service borrowing one route must not close another route's client.
        return

    def __getattr__(self, name: str) -> Any:
        member = getattr(self._owner.client(self._name), name)
        if not callable(member):
            return member
        # The user knows the environment model as the reasoning brain.
        service = "chat" if self._name == "chat" else "reasoning"
        if name in {"stream", "stream_json"}:

            async def stream(*args: Any, **kwargs: Any):
                with self._owner.lease(self._name) as client:
                    async with closing_iterator(
                        getattr(client, name)(*args, **kwargs)
                    ) as items:
                        while True:
                            # A consumer may advance or close this generator in
                            # another task. Never keep a ContextVar token across
                            # yield, or expose the route scope to the consumer.
                            with call_scope(self._name, fallback=True):
                                try:
                                    item = await anext(items)
                                except StopAsyncIteration:
                                    break
                                except Exception as error:
                                    failures.noted(error, service)
                                    raise
                            yield item
                failures.succeeded(service)

            return stream
        if inspect.iscoroutinefunction(member):

            async def invoke(*args: Any, **kwargs: Any):
                with (
                    self._owner.lease(self._name) as client,
                    call_scope(self._name, fallback=True),
                ):
                    try:
                        result = await getattr(client, name)(*args, **kwargs)
                    except Exception as error:
                        failures.noted(error, service)
                        raise
                    failures.succeeded(service)
                    return result

            return invoke

        def invoke_sync(*args: Any, **kwargs: Any):
            with self._owner.lease(self._name) as client:
                return getattr(client, name)(*args, **kwargs)

        return invoke_sync


class ModelRuntime:
    """Own model clients; unchanged endpoints retain their connection pools."""

    def __init__(
        self,
        routes: DesktopModelRoutes,
        *,
        text_factory: Callable[..., Any],
        environment_factory: Callable[..., Any],
    ) -> None:
        self._text_factory = text_factory
        self._environment_factory = environment_factory
        self._current: _Generation | None = None
        self._generations: set[_Generation] = set()
        self._cleanup: set[asyncio.Task[None]] = set()
        self._closed = False
        self.revision = 0
        self.chat = ModelRoute(self, "chat")
        self.reasoning = ModelRoute(self, "reasoning")
        self.environment = ModelRoute(self, "environment")
        self.configure(routes)

    @property
    def routes(self) -> DesktopModelRoutes:
        assert self._current is not None
        return self._current.routes

    def configure(self, routes: DesktopModelRoutes) -> bool:
        if self._closed:
            raise RuntimeError("model runtime is closed")
        previous = self._current
        if previous is not None and previous.routes == routes:
            return False
        reusable: dict[LanguageModelEndpoint, _Client] = {}
        if previous is not None:
            reusable[previous.routes.chat] = previous.chat
            reusable[previous.routes.reasoning] = previous.reasoning
        created: list[_Client] = []

        def text(endpoint: LanguageModelEndpoint) -> _Client:
            if endpoint not in reusable:
                resource = _Client(self._text_factory(**endpoint.client_arguments()))
                reusable[endpoint] = resource
                created.append(resource)
            return reusable[endpoint]

        try:
            chat = text(routes.chat)
            reasoning = text(routes.reasoning)
            environment = _Client(
                self._environment_factory(**routes.reasoning.client_arguments())
            )
            created.append(environment)
        except BaseException:
            for resource in created:
                self._retire(resource)
            raise
        generation = _Generation(routes, chat, reasoning, environment)
        for resource in generation.clients():
            resource.references += 1
        self._generations.add(generation)
        # No awaits between publishing routes: every new operation sees the
        # same complete configuration, never half of a chat/reasoning update.
        self._current = generation
        self.revision += 1
        if previous is not None:
            self._release(previous)
        return True

    def client(self, name: str) -> Any:
        if self._closed or self._current is None:
            raise RuntimeError("model runtime is closed")
        return getattr(self._current, name).value

    @contextmanager
    def lease(self, name: str):
        if self._closed or self._current is None:
            raise RuntimeError("model runtime is closed")
        generation = self._current
        generation.references += 1
        try:
            yield getattr(generation, name).value
        finally:
            self._release(generation)

    def _release(self, generation: _Generation) -> None:
        generation.references -= 1
        if generation.references:
            return
        self._generations.discard(generation)
        for resource in generation.clients():
            resource.references -= 1
            if resource.references == 0:
                self._retire(resource)

    def _retire(self, resource: _Client) -> None:
        async def close() -> None:
            try:
                await resource.value.close()
            except Exception:
                logger.warning(
                    "Retired model connection cleanup failed", exc_info=False
                )

        task = asyncio.create_task(close(), name="retire-model-client")
        self._cleanup.add(task)
        task.add_done_callback(self._cleanup.discard)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._current is not None:
                self._release(self._current)
                self._current = None
        # App shutdown closes request owners first. Remaining leases (if any)
        # still retire themselves when their requests leave the finally block.
        while self._cleanup:
            await asyncio.gather(*tuple(self._cleanup))
