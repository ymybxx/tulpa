import pytest

from fish_memory.performance.tracker import MessagePerformanceTracker


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def test_records_offsets_durations_and_repeated_indices() -> None:
    clock = FakeClock(
        [
            1_000_000,
            1_100_000,
            1_350_000,
            1_500_000,
            1_800_000,
            1_900_000,
        ]
    )
    tracker = MessagePerformanceTracker(clock=clock)

    with tracker.measure("history.load"):
        pass
    with tracker.measure("history.load"):
        pass
    steps = tracker.finish()

    assert [(step.name, step.index) for step in steps] == [
        ("history.load", 0),
        ("history.load", 1),
        ("total", 0),
    ]
    assert steps[0].started_offset_us == 100
    assert steps[0].duration_us == 250
    assert steps[1].started_offset_us == 500
    assert steps[1].duration_us == 300
    assert steps[-1].duration_us == 900


def test_records_payloads_without_serializing_them() -> None:
    tracker = MessagePerformanceTracker(clock=FakeClock([0, 20_000, 50_000]))
    input_payload = {"messages": [{"role": "user", "content": "你好"}]}
    output_payload = {"content": "你好呀"}

    tracker.record(
        "llm.answer",
        started_ns=20_000,
        finished_ns=50_000,
        input_payload=input_payload,
        output_payload=output_payload,
    )

    step = tracker.snapshot()[0]
    assert step.input_payload is input_payload
    assert step.output_payload is output_payload


def test_records_exception_type_and_reraises() -> None:
    tracker = MessagePerformanceTracker(clock=FakeClock([0, 10_000, 20_000, 30_000]))

    with pytest.raises(RuntimeError, match="boom"):
        with tracker.measure("history.load"):
            raise RuntimeError("boom")

    assert tracker.finish()[0].error == "RuntimeError"


def test_records_cancelled_error_and_reraises() -> None:
    tracker = MessagePerformanceTracker(clock=FakeClock([0, 10_000, 20_000, 30_000]))

    with pytest.raises(BaseException) as raised:
        with tracker.measure("llm.answer"):
            raise GeneratorExit

    assert raised.type is GeneratorExit
    assert tracker.finish()[0].error == "GeneratorExit"


def test_finish_returns_tuple_and_rejects_mutation_after_finish() -> None:
    tracker = MessagePerformanceTracker(clock=FakeClock([0, 1_000]))

    steps = tracker.finish()

    assert isinstance(steps, tuple)
    assert steps[0].name == "total"
    with pytest.raises(RuntimeError, match="finished"):
        tracker.finish()
    with pytest.raises(RuntimeError, match="finished"):
        tracker.record("late", started_ns=0, finished_ns=1_000)
