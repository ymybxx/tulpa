import asyncio
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

from pg0 import InstanceInfo, Pg0

DEFAULT_INSTANCE_NAME = "fish-memory"
DEFAULT_USERNAME = "fish_memory"
DEFAULT_PASSWORD = "fish_memory"
DEFAULT_DATABASE = "fish_memory"
_DESKTOP_POSTGRES_CONFIG = {
    "shared_buffers": "64MB",
    "maintenance_work_mem": "64MB",
    "work_mem": "4MB",
    "effective_cache_size": "256MB",
    "max_parallel_maintenance_workers": "1",
    "jit": "off",
}
_MACOS_UNSUPPORTED_LOCALE = "C.UTF-8"
_MACOS_FALLBACK_LOCALE = "en_US.UTF-8"
_LOCALE_ENV_NAMES = ("LANG", "LC_ALL", "LC_CTYPE")
_PG0_START_ENV_LOCK = Lock()
logger = logging.getLogger(__name__)


class Pg0Instance(Protocol):
    def info(self) -> InstanceInfo: ...

    def start(self) -> InstanceInfo: ...

    def stop(self) -> None: ...


class Pg0Factory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        port: int | None,
        username: str,
        password: str,
        database: str,
        data_dir: str | None,
        config: dict[str, str],
    ) -> Pg0Instance: ...


@dataclass(frozen=True)
class Pg0Target:
    name: str
    port: int | None


@dataclass
class ResolvedDatabase:
    database_url: str
    _owned_pg0: Pg0Instance | None = field(default=None, repr=False)

    async def close(self) -> None:
        if self._owned_pg0 is None:
            return
        await asyncio.to_thread(self._owned_pg0.stop)
        self._owned_pg0 = None


def parse_pg0_url(database_url: str) -> Pg0Target | None:
    if database_url == "pg0":
        return Pg0Target(name=DEFAULT_INSTANCE_NAME, port=None)
    if not database_url.startswith("pg0://"):
        return None

    value = database_url.removeprefix("pg0://")
    if ":" not in value:
        return Pg0Target(name=value or DEFAULT_INSTANCE_NAME, port=None)

    name, port_text = value.rsplit(":", 1)
    return Pg0Target(
        name=name or DEFAULT_INSTANCE_NAME,
        port=int(port_text),
    )


def _create_pg0(
    *,
    name: str,
    port: int | None,
    username: str,
    password: str,
    database: str,
    data_dir: str | None,
    config: dict[str, str],
) -> Pg0:
    return Pg0(
        name=name,
        port=port,
        username=username,
        password=password,
        database=database,
        data_dir=data_dir,
        config=config,
    )


@contextmanager
def _pg0_start_environment() -> Iterator[None]:
    if sys.platform != "darwin":
        yield
        return

    # macOS does not provide C.UTF-8 even though many shells export it. pg0
    # delegates to initdb, which rejects that locale on first initialization.
    with _PG0_START_ENV_LOCK:
        original = {name: os.environ.get(name) for name in _LOCALE_ENV_NAMES}
        for name, value in original.items():
            if value == _MACOS_UNSUPPORTED_LOCALE:
                os.environ[name] = _MACOS_FALLBACK_LOCALE
        try:
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _start_pg0(instance: Pg0Instance) -> InstanceInfo:
    with _pg0_start_environment():
        return instance.start()


class _Pg0Startup:
    """Hand ownership to the caller, or stop after an abandoned native start."""

    def __init__(self, instance: Pg0Instance) -> None:
        self.instance = instance
        self._lock = Lock()
        self._finished = False
        self._abandoned = False
        self._cleanup_claimed = False

    def _claim_cleanup(self) -> bool:
        # Caller holds the lock; exactly one thread may own cleanup.
        if self._finished and self._abandoned and not self._cleanup_claimed:
            self._cleanup_claimed = True
            return True
        return False

    def abandon(self) -> bool:
        with self._lock:
            self._abandoned = True
            return self._claim_cleanup()

    def cleanup(self) -> None:
        try:
            self.instance.stop()
        except Exception:
            logger.exception("Unable to stop abandoned embedded PostgreSQL startup")

    def start(self) -> InstanceInfo:
        try:
            return _start_pg0(self.instance)
        except BaseException:
            with self._lock:
                cancelled = self._abandoned
                self._abandoned = True
            if cancelled:
                logger.exception(
                    "Embedded PostgreSQL startup failed after cancellation"
                )
            raise
        finally:
            with self._lock:
                self._finished = True
                cleanup = self._claim_cleanup()
            if cleanup:
                self.cleanup()


async def resolve_database(
    database_url: str,
    *,
    data_dir: str | None = None,
    desktop_mode: bool = False,
    pg0_factory: Pg0Factory | None = None,
) -> ResolvedDatabase:
    target = parse_pg0_url(database_url)
    if target is None:
        return ResolvedDatabase(database_url=database_url)

    factory = pg0_factory or _create_pg0
    instance = factory(
        name=target.name,
        port=target.port,
        username=DEFAULT_USERNAME,
        password=DEFAULT_PASSWORD,
        database=DEFAULT_DATABASE,
        data_dir=data_dir,
        config=dict(_DESKTOP_POSTGRES_CONFIG) if desktop_mode else {},
    )
    info = await asyncio.to_thread(instance.info)
    started_here = not info.running
    if desktop_mode and info.running:
        # A force-quit can leave pg0 alive after the owning API process is
        # gone. Restart the app-private instance so new low-memory settings
        # actually take effect, then adopt its lifecycle for clean shutdown.
        await asyncio.to_thread(instance.stop)
        started_here = True
    if started_here:
        startup = _Pg0Startup(instance)
        try:
            info = await asyncio.to_thread(startup.start)
        except BaseException:
            # Cancelling to_thread cannot interrupt pg0's native subprocess.
            # Revoke ownership now. If start is still running, that worker
            # cleans up when it finishes; cancellation never waits for start.
            if startup.abandon():
                await asyncio.to_thread(startup.cleanup)
            raise

    if not info.uri:
        if started_here:
            await asyncio.to_thread(instance.stop)
        raise RuntimeError("embedded PostgreSQL did not return a connection URL")

    return ResolvedDatabase(
        database_url=info.uri,
        _owned_pg0=instance if started_here else None,
    )
