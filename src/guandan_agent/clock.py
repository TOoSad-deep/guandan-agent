"""Injectable clocks; scheduling belongs to the runtime, not to this module."""

from datetime import datetime, timedelta, timezone
import math
import time
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float:
        """Return seconds on this process's monotonic timeline."""
        ...

    def utc_now(self) -> str:
        """Return a UTC ISO 8601 timestamp for display and records."""
        ...


def _seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("clock seconds must be finite numbers")
    return float(value)


def _iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> str:
        return _iso_utc(datetime.now(timezone.utc))


class ManualClock:
    """A deterministic clock advanced explicitly by local tests; never sleeps."""

    def __init__(
        self,
        monotonic_start: float = 0.0,
        utc_start: str = "2000-01-01T00:00:00Z",
    ) -> None:
        self._monotonic = _seconds(monotonic_start)
        if not isinstance(utc_start, str):
            raise ValueError("utc_start must be a UTC ISO 8601 timestamp")
        self._utc = datetime.fromisoformat(utc_start)
        if "T" not in utc_start or self._utc.utcoffset() != timedelta(0):
            raise ValueError("utc_start must be a UTC ISO 8601 timestamp")
        self._utc = self._utc.astimezone(timezone.utc)

    def monotonic(self) -> float:
        return self._monotonic

    def utc_now(self) -> str:
        return _iso_utc(self._utc)

    def advance(self, seconds: float) -> None:
        step = _seconds(seconds)
        if step < 0:
            raise ValueError("a monotonic clock cannot move backwards")
        new_monotonic = self._monotonic + step
        if not math.isfinite(new_monotonic):
            raise ValueError("clock would overflow")
        new_utc = self._utc + timedelta(seconds=step)
        self._monotonic = new_monotonic
        self._utc = new_utc
