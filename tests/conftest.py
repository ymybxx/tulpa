import sys
import time
from collections.abc import Callable, Iterator
from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def _windows_tzset() -> Callable[[], None] | None:
    """Refresh only this process's CRT timezone, without changing Windows."""
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        refresh = ctypes.CDLL("ucrtbase")._tzset
    except (OSError, AttributeError):
        return None
    refresh.argtypes = []
    refresh.restype = None
    return refresh


@pytest.fixture(autouse=True)
def system_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Prefer UTC where supported without skipping unrelated Windows tests."""
    tzset = getattr(time, "tzset", None)
    windows_tzset = _windows_tzset() if tzset is None else None
    refresh = tzset or windows_tzset
    try:
        with monkeypatch.context() as context:

            def set_timezone(name: str) -> None:
                if tzset is None:
                    if name != "UTC":
                        pytest.skip("this platform cannot switch POSIX TZ timezones")
                    if windows_tzset is None:
                        if time.timezone != 0 or (time.daylight and time.altzone != 0):
                            pytest.skip(
                                "this platform requires an existing UTC system timezone"
                            )
                        return
                    context.setenv("TZ", "UTC0")
                    windows_tzset()
                    # Python initializes these exports when time is imported;
                    # direct CRT _tzset does not refresh Python's cached copies.
                    for attribute in ("timezone", "altzone", "daylight"):
                        context.setattr(time, attribute, 0)
                    context.setattr(time, "tzname", ("UTC", "UTC"))
                    return
                context.setenv("TZ", name)
                tzset()

            # The CRT supports fixed UTC but not IANA zones. Without either
            # refresh API, ordinary tests still execute in the existing zone.
            if refresh is not None:
                set_timezone("UTC")
            yield set_timezone
    finally:
        if refresh is not None:
            refresh()


@pytest.fixture(autouse=True)
def offline_model_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model downloads never reach a real mirror or bundle from the test suite."""
    from fish_memory import model_sources

    def unreachable(url, destination, on_bytes) -> None:
        raise OSError(f"network disabled in tests: {url}")

    monkeypatch.setattr(model_sources, "_stream", unreachable)
    monkeypatch.setattr(model_sources, "_retry_delay", lambda _attempt: 0.0)
    monkeypatch.delenv(model_sources.BUNDLED_MODELS_ENV, raising=False)
    monkeypatch.delenv(model_sources.MODEL_STATUS_ENV, raising=False)
