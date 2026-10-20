"""B07 graph branches use a real LangGraph/Gateway and offline ProviderPort."""

import asyncio
import json

import pytest

from guandan_agent.config import ReviewConfig
from guandan_agent.contracts import BudgetView, CallTicket, GateDecision, ProviderReply, Usage
from guandan_agent.model import ModelGateway, ScriptedProvider, ScriptedStep
from guandan_agent.model.provider import safe_error
from guandan_agent.strategy import DecisionPolicy
from tests.unit.test_model import FakeCalls, model_case
from tests.unit.test_risk import risk_case
from tests.unit.test_rules import UnusableCalls


MAIN = '{"action_id":"L0000","reason":"main choice"}'
KEPT = '{"action_id":"L0000","reason":"review confirms"}'
CHANGED = '{"action_id":"L0001","reason":"review changes"}'


class V2Calls(FakeCalls):
    def __init__(self, binding, *, slots_limit=2, **kwargs):
        super().__init__(binding, **kwargs)
        self.slots_limit = slots_limit
        self.deny_second_begin = False
        self.roles = []

    def status(self, binding):
        return BudgetView(**super().status(binding).model_dump(exclude={"max_slots"}), max_slots=2)

    def admit_call(self, binding, role):
        view = self.status(binding)
        if not view.active or not view.suggestions_open:
            return safe_error("REQUEST_STALE", "Stale request.")
        if self.slots >= self.slots_limit:
            return safe_error("BATCH_BUDGET", "Run call budget exhausted.")
        if role != "main" and view.remaining_s < 2.0:
            return safe_error("NO_REVIEW_TIME", "Insufficient remaining time for a second call.")
        self.slots += 1
        self.roles.append(role)
        return CallTicket(call_id=f"test-call-{self.slots}", binding=binding, role=role,
                          ordinal=self.slots, deadline_mono=self.deadline)

    def begin_call(self, ticket):
        if ticket.ordinal == 2 and self.deny_second_begin:
            return GateDecision(status="deny", error=safe_error("NO_REVIEW_TIME", "Second call denied."))
        return super().begin_call(ticket)


async def execute(steps, *, input=None, configure=None, provider_factory=None):
    input = input or risk_case(counts=(2, 1, None, None))
    calls = V2Calls(input.context.state.binding)
    if configure:
        configure(calls)
    provider = (provider_factory(calls) if provider_factory else
                ScriptedProvider(steps, clock=calls.clock))
    gateway = ModelGateway(provider, clock=calls.clock, prompt_version="b07-review-v1")
    try:
        result = await DecisionPolicy(gateway=gateway, review_config=ReviewConfig(enabled=True)).decide(input, calls)
    finally:
        await gateway.aclose()
    return result, calls, provider


@pytest.mark.parametrize("second,status,action", [(KEPT, "kept", "L0000"), (CHANGED, "changed", "L0001")])
def test_multiple_risks_use_one_review_and_compare_ids_in_program(second, status, action):
    input = risk_case(kind="Bomb", leader=2, counts=(2, 1, 4, 2))
    result, calls, provider = asyncio.run(execute([MAIN, second, MAIN], input=input))
    proposal = result.proposal
    assert proposal.source == "review" and proposal.action_id == action
    assert proposal.review.kind == "review" and proposal.review.status == status
    assert proposal.review.before.action_id == "L0000" and proposal.review.after.action_id == action
    assert proposal.review.trigger_codes == ("OPPONENT_NEAR_FINISH", "USE_BOMB", "BEAT_PARTNER")
    assert calls.registered[0].trigger_codes == proposal.review.trigger_codes
    assert calls.roles == ["main", "review"] and calls.started == len(provider.inputs) == 2
    payloads = [json.loads(item.messages[1].content) for item in provider.inputs]
    assert payloads[0]["candidates"] == payloads[1]["candidates"]
    assert payloads[1]["main"]["action_id"] == "L0000"
    assert proposal.candidate_snapshot_id == input.candidates.candidate_snapshot_id
    assert result.error is None


def test_no_risk_registers_and_returns_main_without_second():
    result, calls, provider = asyncio.run(execute([MAIN, CHANGED], input=model_case(policy="V2")))
    assert result.proposal.source == "main" and result.proposal.review is None
    assert calls.registered[0].trigger_codes == ()
    assert calls.started == len(provider.inputs) == 1


@pytest.mark.parametrize("main,error", [("not json", "MODEL_SCHEMA"),
                                       ('{"action_id":"missing","reason":"x"}', "MODEL_ACTION_UNKNOWN")])
def test_invalid_main_repairs_once_without_later_risk_review(main, error):
    input = risk_case(kind="Bomb", leader=2, counts=(2, 1, 4, 2))
    result, calls, provider = asyncio.run(execute([main, MAIN, CHANGED], input=input))
    assert result.proposal.source == "repair"
    review = result.proposal.review
    assert review.kind == "repair" and review.status == "repaired"
    assert review.before is None and review.after.action_id == "L0000"
    assert review.trigger_codes == (error,) and review.error is None
    assert calls.registered == [] and calls.roles == ["main", "repair"]
    assert len(provider.inputs) == calls.started == 2 and result.error is None
    payload = json.loads(provider.inputs[1].messages[1].content)
    assert payload["validation_error"] == error
    assert payload.get("main") is None


@pytest.mark.parametrize("main,error", [
    ("", "MODEL_EMPTY"), (None, "MODEL_EMPTY"),
    (ScriptedStep(error=safe_error("MODEL_TRANSPORT", "Service unavailable.")), "MODEL_TRANSPORT"),
    (ScriptedStep(error=safe_error("MODEL_TIMEOUT", "Service timed out.")), "MODEL_TIMEOUT"),
])
def test_empty_or_provider_failure_never_uses_repair(main, error):
    result, calls, provider = asyncio.run(execute([main, MAIN]))
    assert result.proposal.source == "fallback" and result.proposal.review is None
    assert result.proposal.cause_code == error and result.error.code == error
    assert calls.registered == [] and calls.roles == ["main"] and len(provider.inputs) == 1


@pytest.mark.parametrize("first,kind", [(MAIN, "review"), ("bad json", "repair")])
@pytest.mark.parametrize("second,error", [
    ("", "MODEL_EMPTY"), ("bad json", "MODEL_SCHEMA"),
    ('{"action_id":"missing","reason":"x"}', "MODEL_ACTION_UNKNOWN"),
    (ScriptedStep(error=safe_error("MODEL_TRANSPORT", "Service unavailable.")), "MODEL_TRANSPORT"),
    (ScriptedStep(error=safe_error("MODEL_TIMEOUT", "Service timed out.")), "MODEL_TIMEOUT"),
])
def test_started_second_failure_falls_back_without_third_call(first, kind, second, error):
    result, calls, provider = asyncio.run(execute([first, second, MAIN]))
    assert result.proposal.source == "fallback" and result.proposal.action_id == "L0000"
    review = result.proposal.review
    assert review.kind == kind and review.status == "failed" and review.after is None
    assert review.error.code == result.error.code == result.proposal.cause_code == error
    assert review.call_id == "test-call-2"
    assert result.proposal.candidate_snapshot_id.endswith(":q1:c1")
    assert calls.roles == ["main", kind] and calls.started == len(provider.inputs) == 2


@pytest.mark.parametrize("first,source,kind", [(MAIN, "main", "review"), ("bad json", "fallback", "repair")])
@pytest.mark.parametrize("denial", ["unreserved", "reserved", "time"])
def test_unstarted_second_skips_keeps_valid_main_or_falls_back(first, source, kind, denial):
    def configure(calls):
        if denial == "unreserved":
            calls.slots_limit = 1
        elif denial == "reserved":
            calls.deny_second_begin = True
        else:
            calls.deadline = 1.5
    input = model_case(policy="V2")
    state = input.context.state.model_copy(update={"remaining_counts": (2, 1, None, None)})
    input = input.model_copy(update={"context": input.context.model_copy(update={"state": state})})
    result, calls, provider = asyncio.run(execute([first, CHANGED], input=input, configure=configure))
    assert result.proposal.source == source
    assert result.proposal.action_id == ("L0000" if source == "main" else "L0001")
    review = result.proposal.review
    assert review.kind == kind and review.status == "skipped" and review.after is None
    assert review.call_id == ("test-call-2" if denial == "reserved" else None)
    assert review.error == result.error
    assert result.proposal.candidate_snapshot_id == input.candidates.candidate_snapshot_id
    assert calls.slots == (2 if denial == "reserved" else 1)
    assert calls.started == len(provider.inputs) == 1


@pytest.mark.parametrize("invalidate_after", [1, 2])
def test_invalidation_during_either_call_never_submits_old_or_fallback(invalidate_after):
    def factory(calls):
        class InvalidatingProvider:
            def __init__(self):
                self.inputs = []
            async def complete(self, input):
                self.inputs.append(input)
                if len(self.inputs) == invalidate_after:
                    calls.active = False
                return ProviderReply(call_id=input.call_id, content=MAIN, usage=Usage(),
                                     transport_requests=0, elapsed_ms=0.0, error=None)
            async def aclose(self):
                pass
        return InvalidatingProvider()
    result, calls, provider = asyncio.run(execute([], provider_factory=factory))
    assert result.status == "invalidated" and result.proposal is None
    assert result.error.code == "REQUEST_STALE" and len(provider.inputs) == invalidate_after
    assert len(calls.registered) == (1 if invalidate_after == 2 else 0)


def test_failed_main_registration_prevents_second_call():
    result, calls, provider = asyncio.run(execute([MAIN, CHANGED], configure=lambda calls: setattr(calls, "allow_registration", False)))
    assert result.status == "invalidated" and result.proposal is None
    assert calls.started == len(provider.inputs) == 1


def test_v2_missing_candidates_uses_fallback_without_call_authority():
    async def scenario():
        provider = ScriptedProvider([])
        input = model_case(policy="V2").model_copy(update={"candidates": None})
        result = await DecisionPolicy(gateway=ModelGateway(provider)).decide(input, UnusableCalls())
        assert result.proposal.source == "fallback" and result.proposal.review is None
        assert provider.inputs == []
    asyncio.run(scenario())
