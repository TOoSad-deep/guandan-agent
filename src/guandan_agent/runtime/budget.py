"""Conservative run-wide slots shared by the isolated seat authorities.

All operations are short synchronous transitions on the application's single
event loop. Reserving a slot is permanent, including a call denied at its final
gate; it never promises that a provider request actually happened.
"""

from __future__ import annotations

from typing import Protocol

from guandan_agent.contracts import (
    BudgetView, CallTicket, ErrorInfo, GateDecision, MainReady, ModelCallResult,
    RequestBinding,
)


class CallPort(Protocol):
    """Only the request-scoped capabilities needed by a model policy."""

    def status(self, binding: RequestBinding) -> BudgetView: ...
    def admit_call(self, binding: RequestBinding, role: str) -> CallTicket | ErrorInfo: ...
    def begin_call(self, ticket: CallTicket) -> GateDecision: ...
    def record_result(self, result: ModelCallResult) -> None: ...
    def register_main(self, main: MainReady) -> bool: ...


class RunCallBudget:
    def __init__(self, max_slots: int) -> None:
        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or max_slots < 0:
            raise ValueError("max_slots must be a nonnegative integer")
        self._max_slots = max_slots
        self._reserved: set[str] = set()
        self._started: set[str] = set()

    @property
    def max_slots(self) -> int:
        return self._max_slots

    @property
    def slots_used(self) -> int:
        return len(self._reserved)

    @property
    def calls_started(self) -> int:
        return len(self._started)

    def reserve(self, call_id: str) -> bool:
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be nonempty")
        if call_id in self._reserved or self.slots_used >= self.max_slots:
            return False
        self._reserved.add(call_id)
        return True

    def begin(self, call_id: str) -> bool:
        if call_id not in self._reserved or call_id in self._started:
            return False
        self._started.add(call_id)
        return True
