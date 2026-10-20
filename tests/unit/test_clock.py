from datetime import datetime, timedelta

import pytest

from guandan_agent.clock import Clock, ManualClock, SystemClock


def test_manual_clock_drives_two_deadlines_without_sleep() -> None:
    clock: Clock = ManualClock(monotonic_start=100.0, utc_start="2026-10-19T02:00:00Z")
    suggest_deadline = clock.monotonic() + 29.0
    final_deadline = clock.monotonic() + 30.0
    clock.advance(29.0)
    assert not clock.monotonic() < suggest_deadline
    assert clock.monotonic() < final_deadline
    assert clock.utc_now() == "2026-10-19T02:00:29Z"
    clock.advance(1.0)
    assert not clock.monotonic() < final_deadline
    assert clock.utc_now() == "2026-10-19T02:00:30Z"


@pytest.mark.parametrize("seconds", [-1, "1", True, float("inf"), float("nan")])
def test_invalid_advance_does_not_mutate_clock(seconds) -> None:
    clock = ManualClock()
    before = clock.monotonic(), clock.utc_now()
    with pytest.raises(ValueError):
        clock.advance(seconds)
    assert (clock.monotonic(), clock.utc_now()) == before


@pytest.mark.parametrize("utc_start", ["2026-10-19", "2026-10-19T00:00:00", "2026-10-19T00:00:00+08:00"])
def test_manual_clock_requires_utc(utc_start: str) -> None:
    with pytest.raises(ValueError):
        ManualClock(utc_start=utc_start)


def test_system_clock_returns_monotonic_seconds_and_utc_timestamp() -> None:
    clock: Clock = SystemClock()
    first = clock.monotonic()
    assert clock.monotonic() >= first
    assert datetime.fromisoformat(clock.utc_now()).utcoffset() == timedelta(0)
