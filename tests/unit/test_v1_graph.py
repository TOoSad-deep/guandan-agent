"""The real LangGraph uses one authorized main call, never a repair loop."""

import asyncio

import pytest

from guandan_agent.model import ModelGateway, ScriptedProvider
from guandan_agent.strategy import DecisionPolicy
from tests.unit.test_model import FakeCalls, model_case
from tests.unit.test_rules import decision_input, make_state_and_legal, UnusableCalls


def run_policy(content, *, registration=True, candidates=None):
    async def scenario():
        input = model_case(candidates=candidates)
        calls = FakeCalls(input.context.state.binding)
        calls.allow_registration = registration
        provider = ScriptedProvider([content, '{"action_id":"L0001","reason":"must not repair"}'], clock=calls.clock)
        policy = DecisionPolicy(gateway=ModelGateway(provider, clock=calls.clock))
        result = await policy.decide(input, calls)
        return result, calls, provider
    return asyncio.run(scenario())


def test_success_records_then_registers_main_and_returns_bound_proposal():
    result, calls, provider = run_policy('{"action_id":"L0000","reason":"  合法主建议  "}')
    assert result.status == "proposed" and result.proposal.source == "main"
    assert result.proposal.action_id == "L0000" and result.proposal.reason == "合法主建议"
    assert result.proposal.candidate_snapshot_id.endswith(":q1:c1")
    assert result.proposal.review is None and result.error is None
    assert len(calls.registered) == len(calls.results) == calls.started == len(provider.inputs) == 1


@pytest.mark.parametrize("content,error", [
    ('{"action_id":"missing","reason":"unknown"}', "MODEL_ACTION_UNKNOWN"),
    ('{"action_id":"L0000","reason":"ok","extra":true}', "MODEL_SCHEMA"),
    ("", "MODEL_EMPTY"),
])
def test_invalid_main_falls_back_without_second_call(content, error):
    result, calls, provider = run_policy(content)
    assert result.proposal.source == "fallback" and result.proposal.action_id == "L0001"
    assert result.error.code == result.proposal.cause_code == error
    assert calls.registered == [] and len(provider.inputs) == calls.slots == 1


def test_registration_failure_discards_main_without_fallback_submission():
    result, calls, provider = run_policy('{"action_id":"L0000","reason":"ok"}', registration=False)
    assert result.status == "invalidated" and result.proposal is None
    assert result.error.code == "REQUEST_STALE" and len(provider.inputs) == 1


def test_shortlist_singleton_is_not_complete_legal_set_singleton():
    result, calls, provider = run_policy('{"action_id":"L0000","reason":"ok"}', candidates=("L0000",))
    assert result.proposal.source == "main" and len(provider.inputs) == calls.started == 1


def test_complete_unique_and_v0_prepared_choice_do_not_access_call_port():
    async def scenario():
        v0 = model_case(policy="V0")
        result = await DecisionPolicy().decide(v0, UnusableCalls())
        assert result.proposal.source == "rule" and result.proposal.action_id == "L0000"
        assert v0.context.fallback_action_id == "L0001"
        state, legal = make_state_and_legal(["S9"], [("Single", "9", ("S9",))])
        result = await DecisionPolicy().decide(decision_input(state, legal, policy="V1"), UnusableCalls())
        assert result.proposal.source == "unique"
    asyncio.run(scenario())


def test_unconfigured_v2_is_unavailable_and_degraded_state_skips_model():
    async def scenario():
        input = model_case(policy="V2")
        provider = ScriptedProvider([])
        policy = DecisionPolicy(gateway=ModelGateway(provider))
        unavailable = await DecisionPolicy().decide(input, UnusableCalls())
        assert unavailable.status == "unavailable" and unavailable.error.code == "POLICY_UNAVAILABLE"
        input = model_case()
        state = input.context.state.model_copy(update={"facts_status": "degraded"})
        input = input.model_copy(update={"context": input.context.model_copy(update={"state": state})})
        result = await policy.decide(input, UnusableCalls())
        assert result.proposal.source == "fallback" and provider.inputs == []
    asyncio.run(scenario())


def test_request_invalidated_while_main_is_running_cannot_submit_fallback():
    async def scenario():
        from guandan_agent.contracts import ProviderReply, Usage
        input = model_case()
        calls = FakeCalls(input.context.state.binding)
        class InvalidatingProvider:
            async def complete(self, input):
                calls.active = False
                return ProviderReply(call_id=input.call_id, content='{"action_id":"L0000","reason":"old"}',
                                     usage=Usage(), transport_requests=0, elapsed_ms=0.0, error=None)
        result = await DecisionPolicy(gateway=ModelGateway(InvalidatingProvider(), clock=calls.clock)).decide(input, calls)
        assert result.status == "invalidated" and result.proposal is None
        assert calls.registered == [] and calls.started == 1
    asyncio.run(scenario())
