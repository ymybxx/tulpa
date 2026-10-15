import asyncio
import os
import sys
from collections.abc import Callable
from threading import Event

import pytest
from pg0 import InstanceInfo

from fish_memory.db.embedded import Pg0Target, parse_pg0_url, resolve_database


class FakePg0:
    def __init__(self, *, running: bool, uri: str | None) -> None:
        self.running = running
        self.uri = uri
        self.start_calls = 0
        self.stop_calls = 0
        self.start_locale: tuple[str | None, str | None, str | None] | None = None

    def info(self) -> InstanceInfo:
        return InstanceInfo(name="fish-memory", running=self.running, uri=self.uri)

    def start(self) -> InstanceInfo:
        self.start_calls += 1
        self.start_locale = (
            os.getenv("LANG"),
            os.getenv("LC_ALL"),
            os.getenv("LC_CTYPE"),
        )
        self.running = True
        return InstanceInfo(name="fish-memory", running=True, uri=self.uri)

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


def factory_for(
    instance: FakePg0,
    *,
    config: dict[str, str] | None = None,
) -> Callable[..., FakePg0]:
    def factory(**kwargs: object) -> FakePg0:
        assert kwargs == {
            "name": "fish-memory",
            "port": None,
            "username": "fish_memory",
            "password": "fish_memory",
            "database": "fish_memory",
            "data_dir": None,
            "config": config or {},
        }
        return instance

    return factory


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("pg0", Pg0Target(name="fish-memory", port=None)),
        ("pg0://pet-db", Pg0Target(name="pet-db", port=None)),
        ("pg0://pet-db:5544", Pg0Target(name="pet-db", port=5544)),
        ("postgresql://localhost/fish", None),
    ],
)
def test_parses_pg0_urls(
    database_url: str,
    expected: Pg0Target | None,
) -> None:
    assert parse_pg0_url(database_url) == expected


@pytest.mark.asyncio
async def test_external_database_url_does_not_create_pg0() -> None:
    def fail_factory(**kwargs: object) -> FakePg0:
        del kwargs
        raise AssertionError("pg0 factory must not be called")

    resolved = await resolve_database(
        "postgresql://db/fish",
        pg0_factory=fail_factory,
    )

    assert resolved.database_url == "postgresql://db/fish"
    await resolved.close()


@pytest.mark.asyncio
async def test_starts_and_stops_owned_pg0_instance() -> None:
    instance = FakePg0(running=False, uri="postgresql://embedded/fish")

    resolved = await resolve_database(
        "pg0://fish-memory",
        pg0_factory=factory_for(instance),
    )

    assert resolved.database_url == "postgresql://embedded/fish"
    assert instance.start_calls == 1
    await resolved.close()
    assert instance.stop_calls == 1


@pytest.mark.asyncio
async def test_desktop_mode_applies_low_memory_pg0_configuration() -> None:
    instance = FakePg0(running=False, uri="postgresql://embedded/fish")
    config = {
        "shared_buffers": "64MB",
        "maintenance_work_mem": "64MB",
        "work_mem": "4MB",
        "effective_cache_size": "256MB",
        "max_parallel_maintenance_workers": "1",
        "jit": "off",
    }

    resolved = await resolve_database(
        "pg0://fish-memory",
        desktop_mode=True,
        pg0_factory=factory_for(instance, config=config),
    )

    await resolved.close()


@pytest.mark.asyncio
async def test_passes_explicit_data_directory_to_pg0() -> None:
    instance = FakePg0(running=False, uri="postgresql://embedded/fish")

    def factory(**kwargs: object) -> FakePg0:
        assert kwargs["data_dir"] == "/private/app-data/postgres"
        return instance

    resolved = await resolve_database(
        "pg0://fish-memory",
        data_dir="/private/app-data/postgres",
        pg0_factory=factory,
    )

    await resolved.close()


@pytest.mark.asyncio
async def test_reuses_running_pg0_without_stopping_it() -> None:
    instance = FakePg0(running=True, uri="postgresql://embedded/fish")

    resolved = await resolve_database(
        "pg0://fish-memory",
        pg0_factory=factory_for(instance),
    )

    assert instance.start_calls == 0
    await resolved.close()
    assert instance.stop_calls == 0


@pytest.mark.asyncio
async def test_desktop_restarts_and_adopts_an_orphaned_pg0_instance() -> None:
    instance = FakePg0(running=True, uri="postgresql://embedded/fish")
    config = {
        "shared_buffers": "64MB",
        "maintenance_work_mem": "64MB",
        "work_mem": "4MB",
        "effective_cache_size": "256MB",
        "max_parallel_maintenance_workers": "1",
        "jit": "off",
    }

    resolved = await resolve_database(
        "pg0://fish-memory",
        desktop_mode=True,
        pg0_factory=factory_for(instance, config=config),
    )

    assert instance.stop_calls == 1
    assert instance.start_calls == 1
    await resolved.close()
    assert instance.stop_calls == 2


@pytest.mark.asyncio
async def test_stops_new_instance_when_pg0_returns_no_url() -> None:
    instance = FakePg0(running=False, uri=None)

    with pytest.raises(RuntimeError, match="did not return a connection URL"):
        await resolve_database(
            "pg0://fish-memory",
            pg0_factory=factory_for(instance),
        )

    assert instance.start_calls == 1
    assert instance.stop_calls == 1


@pytest.mark.asyncio
async def test_failed_start_stops_partial_instance_and_preserves_error() -> None:
    failure = RuntimeError("synthetic startup failure")

    class FailedPg0(FakePg0):
        def start(self):
            super().start()
            raise failure

    instance = FailedPg0(running=False, uri="postgresql://embedded/fish")
    with pytest.raises(RuntimeError) as error:
        await resolve_database("pg0", pg0_factory=factory_for(instance))
    assert error.value is failure
    assert instance.stop_calls == 1
    assert not instance.running


@pytest.mark.asyncio
@pytest.mark.parametrize("start_fails", [False, True])
async def test_cancelled_start_returns_promptly_then_worker_stops_once(start_fails):
    entered, release, stopped = Event(), Event(), Event()

    class DelayedPg0(FakePg0):
        def start(self):
            entered.set()
            assert release.wait(5), "test did not release the fake startup"
            result = super().start()
            if start_fails:
                raise RuntimeError("synthetic delayed failure")
            return result

        def stop(self):
            super().stop()
            stopped.set()

    instance = DelayedPg0(running=False, uri="postgresql://embedded/fish")
    task = asyncio.create_task(
        resolve_database("pg0", pg0_factory=factory_for(instance))
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert not release.is_set()
        assert instance.stop_calls == 0
    finally:
        release.set()
    assert await asyncio.to_thread(stopped.wait, 2)
    assert instance.stop_calls == 1
    assert not instance.running


@pytest.mark.asyncio
async def test_cancelled_handoff_stops_completed_start_once(monkeypatch):
    import fish_memory.db.embedded as embedded

    instance = FakePg0(running=False, uri="postgresql://embedded/fish")
    real_to_thread = asyncio.to_thread

    async def cancel_after_start(function, *args, **kwargs):
        result = await real_to_thread(function, *args, **kwargs)
        if getattr(function, "__name__", "") == "start":
            raise asyncio.CancelledError
        return result

    monkeypatch.setattr(embedded.asyncio, "to_thread", cancel_after_start)
    with pytest.raises(asyncio.CancelledError):
        await resolve_database("pg0", pg0_factory=factory_for(instance))
    assert instance.start_calls == 1
    assert instance.stop_calls == 1
    assert not instance.running


@pytest.mark.asyncio
async def test_abandoned_start_cleanup_failure_is_logged(caplog):
    class FailedPg0(FakePg0):
        def start(self):
            raise RuntimeError("original startup failure")

        def stop(self):
            self.stop_calls += 1
            raise RuntimeError("synthetic stop failure")

    instance = FailedPg0(running=False, uri=None)
    with pytest.raises(RuntimeError, match="original startup failure"):
        await resolve_database("pg0", pg0_factory=factory_for(instance))
    assert instance.stop_calls == 1
    assert "Unable to stop abandoned embedded PostgreSQL startup" in caplog.text
    assert "synthetic stop failure" in caplog.text


@pytest.mark.asyncio
async def test_normalizes_unsupported_macos_locale_only_during_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("macOS locale compatibility test")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")
    instance = FakePg0(running=False, uri="postgresql://embedded/fish")

    await resolve_database(
        "pg0://fish-memory",
        pg0_factory=factory_for(instance),
    )

    assert instance.start_locale == (
        "en_US.UTF-8",
        "en_US.UTF-8",
        "en_US.UTF-8",
    )
    assert os.environ["LANG"] == "C.UTF-8"
    assert os.environ["LC_ALL"] == "C.UTF-8"
    assert os.environ["LC_CTYPE"] == "C.UTF-8"
