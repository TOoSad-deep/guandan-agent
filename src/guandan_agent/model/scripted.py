"""Offline provider with the same raw-response boundary as DeepSeek."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.contracts import ErrorInfo, ProviderInput, ProviderReply, Usage
from guandan_agent.model.provider import safe_error, unknown_usage


@dataclass(frozen=True)
class ScriptedStep:
    content: str | None = None
    usage: Usage | None = None
    error: ErrorInfo | None = None
    delay_s: float = 0.0


class ScriptedProvider:
    def __init__(self, steps: Sequence[ScriptedStep | str | None], *, clock: Clock | None = None):
        self._steps = tuple(steps)
        self._clock = clock or SystemClock()
        self.inputs: list[ProviderInput] = []
        self.closed = False

    async def complete(self, input: ProviderInput) -> ProviderReply:
        start = self._clock.monotonic()
        remaining = input.deadline_mono - start
        if remaining <= 0:
            return ProviderReply(
                call_id=input.call_id, content=None, usage=unknown_usage(), transport_requests=0,
                elapsed_ms=0.0, error=safe_error("MODEL_TIMEOUT", "Provider deadline has expired."),
            )
        index = len(self.inputs)
        self.inputs.append(input)
        if self.closed or index >= len(self._steps):
            step = ScriptedStep(error=safe_error("MODEL_TRANSPORT", "Scripted provider is unavailable."))
        else:
            scripted = self._steps[index]
            step = scripted if isinstance(scripted, ScriptedStep) else ScriptedStep(content=scripted)
        try:
            async with asyncio.timeout(remaining):
                if step.delay_s:
                    await asyncio.sleep(step.delay_s)
            if self._clock.monotonic() >= input.deadline_mono:
                raise TimeoutError
        except TimeoutError:
            step = ScriptedStep(error=safe_error("MODEL_TIMEOUT", "Provider request exceeded its time limit."))
        return ProviderReply(
            call_id=input.call_id, content=step.content,
            usage=step.usage if step.usage is not None else unknown_usage(),
            transport_requests=0, elapsed_ms=max(0.0, self._clock.monotonic() - start) * 1000,
            error=step.error,
        )

    async def aclose(self) -> None:
        self.closed = True
