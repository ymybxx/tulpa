"""The suite must still execute on a non-UTC Windows machine."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def timezone_patch():
    # Restore simulated TZ changes before the real autouse timezone fixture exits.
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def timezone_fixture(timezone_patch, clock, windows_tzset=None):
    spec = importlib.util.spec_from_file_location(
        "test_environment_fixture", Path(__file__).with_name("conftest.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.time = clock
    module._windows_tzset = lambda: windows_tzset
    return module.system_timezone.__wrapped__(timezone_patch)


def test_non_utc_windows_runs_tests_without_changing_timezone(timezone_patch):
    timezone_patch.delenv("TZ", raising=False)
    fixture = timezone_fixture(
        timezone_patch, SimpleNamespace(timezone=-28800, daylight=0, altzone=-28800)
    )
    setter = next(fixture)
    assert callable(setter)
    with pytest.raises(pytest.skip.Exception, match="existing UTC"):
        setter("UTC")
    with pytest.raises(pytest.skip.Exception, match="cannot switch"):
        setter("America/New_York")
    with pytest.raises(StopIteration):
        next(fixture)
    assert "TZ" not in os.environ


def test_utc_windows_can_run_explicit_utc_tests(timezone_patch):
    fixture = timezone_fixture(
        timezone_patch, SimpleNamespace(timezone=0, daylight=0, altzone=0)
    )
    setter = next(fixture)
    setter("UTC")
    with pytest.raises(StopIteration):
        next(fixture)


def test_posix_timezone_is_restored_after_test(timezone_patch):
    seen = []
    timezone_patch.setenv("TZ", "original-zone")
    fixture = timezone_fixture(
        timezone_patch, SimpleNamespace(tzset=lambda: seen.append(os.environ["TZ"]))
    )
    setter = next(fixture)
    setter("Asia/Shanghai")
    with pytest.raises(StopIteration):
        next(fixture)
    assert seen == ["UTC", "Asia/Shanghai", "original-zone"]


def test_windows_crt_defaults_to_utc_and_restores_cached_time_exports(timezone_patch):
    seen = []
    timezone_patch.setenv("TZ", "original-zone")
    clock = SimpleNamespace(
        timezone=-28800, daylight=0, altzone=-32400, tzname=("standard", "summer")
    )
    fixture = timezone_fixture(
        timezone_patch, clock, windows_tzset=lambda: seen.append(os.environ["TZ"])
    )
    setter = next(fixture)
    assert os.environ["TZ"] == "UTC0"
    assert (clock.timezone, clock.altzone, clock.daylight) == (0, 0, 0)
    assert clock.tzname == ("UTC", "UTC")
    setter("UTC")
    with pytest.raises(pytest.skip.Exception, match="cannot switch"):
        setter("America/New_York")
    with pytest.raises(StopIteration):
        next(fixture)
    assert seen == ["UTC0", "UTC0", "original-zone"]
    assert (clock.timezone, clock.altzone) == (-28800, -32400)
    assert clock.tzname == ("standard", "summer")


@pytest.mark.skipif(sys.platform != "win32", reason="requires the Windows CRT")
def test_real_windows_crt_timezone_is_process_local_and_restored():
    code = """
import json
from datetime import UTC, datetime
import pytest
from tests.conftest import system_timezone
instant = datetime(2026, 9, 3, tzinfo=UTC)
before = instant.astimezone().isoformat()
with pytest.MonkeyPatch.context() as patch:
    fixture = system_timezone.__wrapped__(patch)
    setter = next(fixture)
    during = instant.astimezone().isoformat()
    setter('UTC')
    try:
        next(fixture)
    except StopIteration:
        pass
after = instant.astimezone().isoformat()
print(json.dumps({'before': before, 'during': during, 'after': after}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={name: value for name, value in os.environ.items() if name != "TZ"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["during"] == "2026-09-03T00:00:00+00:00"
    assert report["after"] == report["before"]
