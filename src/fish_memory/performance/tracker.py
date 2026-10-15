import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from fish_memory.performance.models import JsonValue, PerformanceStep


class MessagePerformanceTracker:
    def __init__(
        self,
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
        origin_ns: int | None = None,
    ) -> None:
        self._clock = clock
        self._origin_ns = clock() if origin_ns is None else origin_ns
        self._steps: list[PerformanceStep] = []
        self._counts: dict[str, int] = {}
        self._finished = False

    @property
    def origin_ns(self) -> int:
        return self._origin_ns

    def now_ns(self) -> int:
        return self._clock()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_ns = self.now_ns()
        error: str | None = None
        try:
            yield
        except BaseException as exception:
            error = type(exception).__name__
            raise
        finally:
            self.record(
                name,
                started_ns=started_ns,
                finished_ns=self.now_ns(),
                error=error,
            )

    def record(
        self,
        name: str,
        *,
        started_ns: int,
        finished_ns: int,
        input_payload: JsonValue = None,
        output_payload: JsonValue = None,
        error: str | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError("performance tracker is already finished")
        if started_ns < self._origin_ns or finished_ns < started_ns:
            raise ValueError("performance timestamps must be monotonic")

        index = self._counts.get(name, 0)
        self._counts[name] = index + 1
        self._steps.append(
            PerformanceStep(
                name=name,
                index=index,
                started_offset_us=(started_ns - self._origin_ns) // 1_000,
                duration_us=(finished_ns - started_ns) // 1_000,
                input_payload=input_payload,
                output_payload=output_payload,
                error=error,
            )
        )

    def snapshot(self) -> tuple[PerformanceStep, ...]:
        return tuple(self._steps)

    def finish(self) -> tuple[PerformanceStep, ...]:
        if self._finished:
            raise RuntimeError("performance tracker is already finished")
        finished_ns = self.now_ns()
        self.record(
            "total",
            started_ns=self._origin_ns,
            finished_ns=finished_ns,
        )
        self._finished = True
        return self.snapshot()
