"""Run slots are conservative and shared without asynchronous races."""

import pytest

from guandan_agent.runtime.budget import RunCallBudget


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_budget_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        RunCallBudget(value)


def test_run_slots_are_never_refunded_and_start_requires_reserved_unique_id():
    budget = RunCallBudget(2)
    assert not budget.begin("a")
    assert budget.reserve("a")
    assert not budget.reserve("a")
    assert budget.reserve("b")
    assert not budget.reserve("c")
    assert budget.slots_used == 2 and budget.calls_started == 0
    assert budget.begin("a")
    assert not budget.begin("a")
    assert budget.calls_started == 1
    assert budget.begin("b")
    assert budget.calls_started == 2


def test_zero_run_budget_blocks_every_reservation():
    budget = RunCallBudget(0)
    assert not budget.reserve("seat-0:q1:m1")
    assert not budget.reserve("seat-1:q1:m1")
    assert budget.slots_used == budget.calls_started == 0


@pytest.mark.parametrize("call_id", ["", "  ", None, 4])
def test_empty_call_identifiers_do_not_consume_slots(call_id):
    budget = RunCallBudget(1)
    with pytest.raises(ValueError):
        budget.reserve(call_id)
    assert budget.slots_used == 0
