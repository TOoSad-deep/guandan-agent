"""Bounded first-play analysis of one player's residual hand only.

The official Java worker supplies possible residual groups, never platform
legality. One instance belongs to one run and is never restarted after failure.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.contracts import WireAction


class EngineError(Exception):
    """Safe, classified local failure; raw worker output never leaves this module."""

    def __init__(self, code: str = "ENGINE_UNAVAILABLE") -> None:
        self.code = code
        super().__init__(code)


class ResidualEngine(Protocol):
    async def analyze(
        self, hand: tuple[str, ...], level_rank: str, deadline_mono: float,
    ) -> tuple[WireAction, ...]: ...

    async def aclose(self) -> None: ...


_ACTIONS = TypeAdapter(tuple[WireAction, ...])


class JavaActionEngine:
    def __init__(
        self, jar_path: str | Path, *, java_executable: str = "java",
        clock: Clock | None = None, cleanup_timeout_s: float = 0.2,
    ) -> None:
        if cleanup_timeout_s <= 0:
            raise ValueError("cleanup timeout must be positive")
        self.jar_path = Path(jar_path)
        self.java_executable = java_executable
        self.clock = clock or SystemClock()
        self.cleanup_timeout_s = cleanup_timeout_s
        self.disabled = False
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._request_id = 0
        self._cleanup: asyncio.Task[None] | None = None

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.clock.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def _disable(self) -> None:
        self.disabled = True
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._stop())

    async def _stop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), self.cleanup_timeout_s)
            except TimeoutError:
                if process.returncode is None:
                    process.kill()
                try:
                    await asyncio.wait_for(process.wait(), self.cleanup_timeout_s)
                except TimeoutError:
                    pass
        except (ProcessLookupError, BrokenPipeError, ConnectionResetError):
            pass

    async def aclose(self) -> None:
        self._disable()
        if self._cleanup is not None:
            await asyncio.shield(self._cleanup)

    async def analyze(
        self, hand: tuple[str, ...], level_rank: str, deadline_mono: float,
    ) -> tuple[WireAction, ...]:
        if self.disabled:
            raise EngineError()
        try:
            async with asyncio.timeout(self._remaining(deadline_mono)):
                async with self._lock:
                    self._remaining(deadline_mono)
                    if self.disabled:
                        raise EngineError()
                    if not hand:
                        return ()
                    if self._process is None:
                        self._process = await asyncio.create_subprocess_exec(
                            self.java_executable, "-jar", str(self.jar_path), "--worker",
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL, limit=2 * 1024 * 1024,
                        )
                    process = self._process
                    if self.disabled or process.returncode is not None:
                        raise EngineError()
                    assert process.stdin is not None and process.stdout is not None
                    self._request_id += 1
                    payload = {
                        "id": self._request_id, "mode": "first", "handCards": list(hand),
                        "heartsNum": hand.count(f"H{level_rank}"), "currentRank": level_rank,
                    }
                    process.stdin.write(json.dumps(payload).encode() + b"\n")
                    await process.stdin.drain()
                    line = await process.stdout.readline()
                    self._remaining(deadline_mono)
                    reply = json.loads(line)
                    if (not isinstance(reply, dict) or reply.get("id") != self._request_id
                            or reply.get("ok") is not True):
                        raise EngineError()
                    actions = _ACTIONS.validate_python(tuple(reply["actions"]))
                    own = Counter(hand)
                    if not actions or any(
                        action[0] == "PASS" or Counter(action[2]) - own for action in actions
                    ):
                        raise EngineError()
                    return actions
        except TimeoutError:
            self._disable()
            raise EngineError("TOOL_TIMEOUT") from None
        except asyncio.CancelledError:
            self._disable()
            raise
        except Exception:
            self._disable()
            raise EngineError() from None
