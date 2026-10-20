import asyncio

import pytest

from guandan_agent.clock import ManualClock, SystemClock
from guandan_agent.runtime.scheduling import AsyncioScheduler, ManualScheduler


def test_virtual_timers_share_clock_and_cancel_without_sleeping():
    clock = ManualClock(100)
    scheduler = ManualScheduler(clock)
    observed = []
    scheduler.call_at(103, lambda: observed.append(("first", clock.monotonic())))
    removed = scheduler.call_at(103, lambda: observed.append(("cancelled", clock.monotonic())))
    scheduler.call_at(103, lambda: observed.append(("last", clock.monotonic())))
    removed.cancel()
    assert scheduler.advance(2) == 0
    assert scheduler.advance(1) == 2
    assert observed == [("first", 103), ("last", 103)]
    assert scheduler.run_due() == 0


def test_due_timer_is_not_run_reentrantly_when_registered():
    scheduler = ManualScheduler(ManualClock(10))
    observed = []
    scheduler.call_at(9, lambda: observed.append("done"))
    assert observed == []
    assert scheduler.run_due() == 1


@pytest.mark.parametrize("when", [True, "10", float("inf"), float("nan")])
def test_invalid_deadlines_are_rejected(when):
    with pytest.raises(ValueError):
        ManualScheduler(ManualClock()).call_at(when, lambda: None)


def test_real_loop_uses_matching_monotonic_time():
    async def exercise():
        clock = SystemClock()
        scheduler = AsyncioScheduler(clock)
        done = asyncio.Event()
        scheduler.call_at(clock.monotonic(), done.set)
        await asyncio.wait_for(done.wait(), timeout=0.5)
    asyncio.run(exercise())
