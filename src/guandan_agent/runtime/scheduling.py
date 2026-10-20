"""Small timer adapters over the same monotonic clock used by Session gates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Protocol

from guandan_agent.clock import Clock, ManualClock


class TimerHandle(Protocol):
    def cancel(self) -> None: ...


class Scheduler(Protocol):
    def call_at(self, when: float, callback: Callable[[], None]) -> TimerHandle: ...


def _deadline(when: float) -> float:
    if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(when):
        raise ValueError("timer deadline must be a finite number")
    return float(when)


class AsyncioScheduler:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def call_at(self, when: float, callback: Callable[[], None]) -> TimerHandle:
        delay = max(0.0, _deadline(when) - self.clock.monotonic())
        return asyncio.get_running_loop().call_later(delay, callback)


@dataclass
class _ManualHandle:
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class ManualScheduler:
    """Test scheduler: advance explicitly, without sleeps or background work."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self._entries: list[tuple[float, int, _ManualHandle]] = []
        self._order = itertools.count()

    def call_at(self, when: float, callback: Callable[[], None]) -> TimerHandle:
        handle = _ManualHandle(callback)
        heapq.heappush(self._entries, (_deadline(when), next(self._order), handle))
        return handle

    def run_due(self) -> int:
        count = 0
        while self._entries and self._entries[0][0] <= self.clock.monotonic():
            _, _, handle = heapq.heappop(self._entries)
            if not handle.cancelled:
                handle.cancelled = True
                handle.callback()
                count += 1
        return count

    def advance(self, seconds: float) -> int:
        self.clock.advance(seconds)
        return self.run_due()
