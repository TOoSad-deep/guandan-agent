"""V2 final authority and deadline races with main distinct from fallback."""

import asyncio
import json

import pytest

from guandan_agent.contracts import (
    CallTicket, DecisionOutcome, DecisionProposal, MainReady, ModelDecision,
    ProposalReady, ProviderReply, ReviewResult, Usage,
)
from guandan_agent.model import ModelGateway, ScriptedProvider
from guandan_agent.runtime.budget import RunCallBudget
from guandan_agent.strategy import DecisionPolicy
from test_session import play, request
from test_session_model import ACTIONS, HAND, act, error, make, result


def main_ready(h, ticket, *, triggers=("OPPONENT_NEAR_FINISH",)):
    reply = result(ticket)
    h.session.call_port.record_result(reply)
    main = MainReady(binding=ticket.binding, candidate_snapshot_id=reply.candidate_snapshot_id,
                     call_id=ticket.call_id, decision=reply.decision, trigger_codes=triggers)
    assert h.session.call_port.register_main(main)
    return main


async def prepared(*, kind="review", budget=None):
    h = make(policy_name="V2", budget=budget)
    binding = act(h)
    await h.tick()
    first = h.session.call_port.admit_call(binding, "main")
    assert isinstance(first, CallTicket)
    assert h.session.call_port.begin_call(first).status == "allow"
    main = None
    if kind == "review":
        main = main_ready(h, first)
    else:
        h.session.call_port.record_result(result(first, decision=False))
    return h, binding, first, main


def review_events(h):
    return [event for event in h.recorder.events if event.event_type == "review_finished"]


def final_review(h):
    events = review_events(h)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["binding"]["request_id"] == events[0].request_id
    assert payload["candidate_snapshot_id"] == events[0].request_id + ":c1"
    return ReviewResult.model_validate_json(json.dumps(
        events[0].model_dump(mode="json")["payload"]["review"]))


def proposal_for(h, main, second, reply, *, kind="review"):
    decision = reply.decision
    status = "repaired" if kind == "repair" else (
        "kept" if main.decision.action_id == decision.action_id else "changed")
    review = ReviewResult(kind=kind, status=status,
                          trigger_codes=main.trigger_codes if main else ("MODEL_SCHEMA",),
                          before=main.decision if main else None, after=decision,
                          call_id=second.call_id, error=None)
    return DecisionProposal(binding=second.binding,
                            candidate_snapshot_id=reply.candidate_snapshot_id,
                            action_id=decision.action_id, source=kind,
                            reason=decision.reason, cause_code=None, review=review)


def post_proposal(h, proposal):
    # In-process producers can use model_copy/model_construct. Session still
    # checks authoritative results even when DTO construction was bypassed.
    outcome = DecisionOutcome.model_construct(status="proposed", proposal=proposal, error=None)
    h.session.post(ProposalReady.model_construct(binding=proposal.binding, outcome=outcome))


@pytest.mark.parametrize("stage", ["waiting", "reserved", "started"])
def test_t10_review_watchdog_uses_main_until_second_actually_begins(stage):
    async def scenario():
        budget = RunCallBudget(2)
        h, binding, first, main = await prepared(budget=budget)
        assert main.decision.action_id == "L0001"
        assert h.session.active_context.fallback_action_id == "L0000"
        second = None
        if stage != "waiting":
            second = h.session.call_port.admit_call(binding, "review")
            assert isinstance(second, CallTicket)
            if stage == "started":
                assert h.session.call_port.begin_call(second).status == "allow"
        h.scheduler.advance(29.0)
        await h.tick()
        record = h.session.execution_records[0]
        assert (record.source, record.action_id) == (
            ("fallback", "L0000") if stage == "started" else ("main", "L0001"))
        review = final_review(h)
        assert review.status == ("failed" if stage == "started" else "skipped")
        assert review.before == main.decision
        assert review.call_id == (None if second is None else second.call_id)
        assert budget.slots_used == (1 if stage == "waiting" else 2)
        assert budget.calls_started == (2 if stage == "started" else 1)
        if second is not None:
            assert h.session.call_port.begin_call(second).status == "deny"
        assert not h.session.call_port.register_main(main)
        assert h.session.call_port.admit_call(binding, "repair").code == "DEADLINE"
        h.scheduler.advance(0.0)
        assert len(review_events(h)) == 1 and len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["waiting", "reserved", "started"])
def test_t10_repair_deadline_always_counts_fallback(stage):
    async def scenario():
        h, binding, _, _ = await prepared(kind="repair")
        second = None
        if stage != "waiting":
            second = h.session.call_port.admit_call(binding, "repair")
            if stage == "started":
                assert h.session.call_port.begin_call(second).status == "allow"
        h.scheduler.advance(29.0)
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.execution_records[0].action_id == "L0000"
        review = final_review(h)
        assert review.kind == "repair" and review.before is None
        assert review.status == ("failed" if stage == "started" else "skipped")
        assert review.trigger_codes == ("MODEL_SCHEMA",)
        assert review.call_id == (None if second is None else second.call_id)
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["ordinary", "MODEL_TIMEOUT", "MODEL_EMPTY", "MODEL_TRANSPORT"])
def test_deadline_does_not_invent_review_for_ordinary_main_or_service_failure(mode):
    async def scenario():
        h = make(policy_name="V2")
        binding = act(h)
        await h.tick()
        first = h.session.call_port.admit_call(binding, "main")
        h.session.call_port.begin_call(first)
        if mode == "ordinary":
            main_ready(h, first, triggers=())
        else:
            h.session.call_port.record_result(result(first, decision=False, code=mode))
            assert h.session.call_port.admit_call(binding, "repair").code == "CALL_ORDER"
        h.scheduler.advance(29.0)
        await h.tick()
        assert not review_events(h)
        assert h.session.execution_records[0].source == ("main" if mode == "ordinary" else "fallback")
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("kind,action_id,status", [
    ("review", "L0001", "kept"), ("review", "L0000", "changed"),
    ("repair", "L0001", "repaired"),
])
def test_second_success_is_bound_to_recorded_result_and_confirmed_once(kind, action_id, status):
    async def scenario():
        h, binding, _, main = await prepared(kind=kind)
        second = h.session.call_port.admit_call(binding, kind)
        assert h.session.call_port.begin_call(second).status == "allow"
        reply = result(second).model_copy(update={
            "decision": ModelDecision(action_id=action_id, reason="Second decision")})
        h.session.call_port.record_result(reply)
        proposal = proposal_for(h, main, second, reply, kind=kind)
        post_proposal(h, proposal)
        post_proposal(h, proposal)
        await h.tick()
        assert h.session.execution_records[0].source == kind
        assert final_review(h).status == status
        h.post(play(0, ACTIONS[0 if action_id == "L0000" else 1]))
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1 and len(review_events(h)) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("tamper", ["snapshot", "call_id", "before", "after", "triggers", "source", "result_role", "result_snapshot", "unrecorded"])
def test_forged_second_proposals_cannot_claim_a_success(tamper):
    async def scenario():
        h, binding, _, main = await prepared()
        second = h.session.call_port.admit_call(binding, "review")
        h.session.call_port.begin_call(second)
        reply = result(second)
        if tamper != "unrecorded":
            recorded = reply
            if tamper == "result_role":
                recorded = reply.model_copy(update={"role": "repair"})
            if tamper == "result_snapshot":
                recorded = reply.model_copy(update={"candidate_snapshot_id": "wrong:c1"})
            h.session.call_port.record_result(recorded)
        proposal = proposal_for(h, main, second, reply)
        if tamper == "snapshot":
            proposal = proposal.model_copy(update={"candidate_snapshot_id": "wrong:c1"})
        elif tamper == "source":
            proposal = proposal.model_copy(update={"source": "repair"})
        elif tamper in {"call_id", "before", "after", "triggers"}:
            changes = {
                "call_id": {"call_id": "wrong:m2"},
                "before": {"before": ModelDecision(action_id="L0000", reason="Forged before")},
                "after": {"after": ModelDecision(action_id="L0000", reason="Forged after")},
                "triggers": {"trigger_codes": ("USE_BOMB",)},
            }[tamper]
            proposal = proposal.model_copy(update={"review": proposal.review.model_copy(update=changes)})
        post_proposal(h, proposal)
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.execution_records[0].action_id == "L0000"
        assert final_review(h).status == "failed"
        await h.session.shutdown()
    asyncio.run(scenario())


def test_fake_main_after_second_started_and_third_call_are_denied():
    async def scenario():
        h, binding, _, main = await prepared()
        second = h.session.call_port.admit_call(binding, "review")
        assert h.session.call_port.begin_call(second).status == "allow"
        h.session.call_port.record_result(result(second, decision=False))
        assert h.session.call_port.admit_call(binding, "repair").code == "CALL_BUDGET"
        assert h.session.call_port.begin_call(second).status == "deny"
        post_proposal(h, DecisionProposal(binding=binding, candidate_snapshot_id=main.candidate_snapshot_id,
                                         action_id=main.decision.action_id, source="main",
                                         reason=main.decision.reason, cause_code=None, review=None))
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert final_review(h).status == "failed"
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["review", "repair"])
def test_t12_second_ticket_rechecks_minimum_remaining_time_without_refunding(kind):
    async def scenario():
        budget = RunCallBudget(2)
        h, binding, _, main = await prepared(kind=kind, budget=budget)
        second = h.session.call_port.admit_call(binding, kind)
        h.scheduler.advance(28.0)
        assert h.session.call_port.status(binding).suggestions_open
        assert h.session.call_port.begin_call(second).status == "deny"
        assert budget.slots_used == 2 and budget.calls_started == 1
        assert h.session.call_port.admit_call(binding, kind).code == "CALL_BUDGET"
        h.session.call_port.record_result(result(second, decision=False, code="CALL_DENIED", started=0))
        h.scheduler.advance(1.0)
        await h.tick()
        review = final_review(h)
        assert review.status == "skipped" and review.call_id == second.call_id
        assert review.error.code == "CALL_DENIED"
        assert h.session.execution_records[0].source == ("main" if kind == "review" else "fallback")
        assert h.session.execution_records[0].action_id == (main.decision.action_id if main else "L0000")
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["stop", "public_fact", "final_deadline"])
def test_t04_t12_second_success_reservation_cannot_cross_write_gate_after_invalidation(change):
    async def scenario():
        h, binding, _, main = await prepared()
        h.platform.before_gate.clear()
        second = h.session.call_port.admit_call(binding, "review")
        h.session.call_port.begin_call(second)
        reply = result(second)
        h.session.call_port.record_result(reply)
        proposal = proposal_for(h, main, second, reply)
        post_proposal(h, proposal)
        await h.tick()
        assert h.session.execution_records[0].status == "reserved"
        if change == "stop":
            h.session.stop()
        elif change == "public_fact":
            h.post(play(1, ["Single", "4", ["S4"]]))
        else:
            h.scheduler.advance(30.0)
        h.platform.before_gate.set()
        post_proposal(h, proposal)
        await h.tick()
        assert not h.platform.written
        assert h.session.execution_records[0].status == "cancelled_before_write"
        assert len(review_events(h)) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["stop", "new_request"])
def test_t13_late_second_result_never_revives_old_request(change):
    async def scenario():
        h, binding, _, main = await prepared()
        second = h.session.call_port.admit_call(binding, "review")
        h.session.call_port.begin_call(second)
        if change == "stop":
            h.session.stop()
        else:
            action = ["Single", "4", ["S4"]]
            h.post(play(1, action))
            h.post(request(hand=HAND, actions=ACTIONS, counts=[2, 26, 27, 27],
                           context=(1, action), leader=(1, action)))
        late = result(second).model_copy(update={"usage": Usage(total_tokens=11)})
        h.session.call_port.record_result(late)
        post_proposal(h, proposal_for(h, main, second, late))
        assert not h.session.call_port.register_main(main)
        assert h.session.call_port.begin_call(second).status == "deny"
        await h.tick()
        assert not h.platform.written and not review_events(h)
        assert h.session.call_results[-1].usage.total_tokens == 11
        if change == "new_request":
            current = h.session.active_context.state.binding
            assert current != binding
            assert h.session.call_port.status(current).slots_used == 0
            assert h.session.registered_main is None
        await h.session.shutdown()
        before = len(h.recorder.events)
        h.session.call_port.record_result(late.model_copy(update={"usage": Usage(total_tokens=99)}))
        assert len(h.recorder.events) == before
    asyncio.run(scenario())


MAIN = '{"action_id":"L0001","reason":"Preserve the low card"}'
CHANGED = '{"action_id":"L0000","reason":"Review selected the low card"}'


async def settle_until(predicate):
    for _ in range(250):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "cooperative work did not reach the expected boundary"


def risky_act(h):
    h.post(request(hand=HAND, actions=ACTIONS, counts=[2, 1, 27, 27]))
    return h.session.active_context.state.binding


@pytest.mark.parametrize("steps,budget,source,status", [
    ([MAIN, MAIN], 2, "review", "kept"),
    ([MAIN, CHANGED], 2, "review", "changed"),
    (["not-json", MAIN], 2, "repair", "repaired"),
    ([MAIN, "not-json"], 2, "fallback", "failed"),
    (["not-json", "not-json"], 2, "fallback", "failed"),
    ([""], 2, "fallback", None),
    ([MAIN], 1, "main", "skipped"),
    (["not-json"], 1, "fallback", "skipped"),
])
def test_t03_actual_graph_gateway_session_second_paths(steps, budget, source, status):
    async def scenario():
        h = make(policy_name="V2", budget=RunCallBudget(budget))
        provider = ScriptedProvider(steps, clock=h.clock)
        gateway = ModelGateway(provider, clock=h.clock)
        h.session.policy = DecisionPolicy(gateway=gateway, review_config=h.session.config.review)
        binding = risky_act(h)
        await settle_until(lambda: bool(h.platform.written))
        record = h.session.execution_records[0]
        assert record.source == source
        assert len(provider.inputs) == min(len(steps), budget)
        assert h.session.call_port.status(binding).slots_used <= 2
        if status is None:
            assert not review_events(h)
        else:
            review = final_review(h)
            assert review.status == status
            assert review.kind == ("repair" if steps[0] == "not-json" else "review")
            if status == "skipped":
                assert review.call_id is None and review.error.code == "BATCH_BUDGET"
        h.post(play(0, ACTIONS[0 if record.action_id == "L0000" else 1]))
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1
        await h.session.shutdown()
        await gateway.aclose()
    asyncio.run(scenario())


def test_t10_real_gateway_waits_for_semaphore_without_reserving_review_ticket():
    class OccupiedGateway(ModelGateway):
        async def choose(self, input, calls):
            reply = await super().choose(input, calls)
            if input.role == "main":
                # Deterministically emulate another seat taking the shared
                # permit between this main reply and its review node.
                await self._semaphore.acquire()
            return reply

    async def scenario():
        budget = RunCallBudget(2)
        h = make(policy_name="V2", budget=budget)
        provider = ScriptedProvider([MAIN, CHANGED], clock=h.clock)
        gateway = OccupiedGateway(provider, clock=h.clock)
        h.session.policy = DecisionPolicy(gateway=gateway, review_config=h.session.config.review)
        binding = risky_act(h)
        await settle_until(lambda: h.session.registered_main is not None)
        await h.tick()
        assert len(provider.inputs) == 1 and budget.slots_used == 1
        assert h.session.registered_main.decision.action_id == "L0001"
        h.scheduler.advance(29.0)
        await h.tick()
        assert h.session.execution_records[0].source == "main"
        assert h.session.execution_records[0].action_id == "L0001"
        review = final_review(h)
        assert review.status == "skipped" and review.call_id is None
        gateway._semaphore.release()
        await h.tick()
        assert len(provider.inputs) == 1 and budget.slots_used == 1
        assert not h.session.call_port.status(binding).suggestions_open
        assert len(h.platform.written) == 1 and len(review_events(h)) == 1
        await h.session.shutdown()
        await gateway.aclose()
    asyncio.run(scenario())


def test_t04_t13_cancellation_resistant_second_only_enriches_usage_after_watchdog():
    class ResistantProvider:
        def __init__(self):
            self.inputs = []
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def complete(self, input):
            self.inputs.append(input)
            if len(self.inputs) == 1:
                return ProviderReply(call_id=input.call_id, content=MAIN, usage=Usage(),
                                     transport_requests=0, elapsed_ms=0.0, error=None)
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
            return ProviderReply(call_id=input.call_id, content=MAIN,
                                 usage=Usage(total_tokens=17), transport_requests=0,
                                 elapsed_ms=0.0, error=None)

        async def aclose(self):
            self.release.set()

    async def scenario():
        h = make(policy_name="V2", budget=RunCallBudget(2))
        provider = ResistantProvider()
        gateway = ModelGateway(provider, clock=h.clock)
        h.session.policy = DecisionPolicy(gateway=gateway, review_config=h.session.config.review)
        binding = risky_act(h)
        await settle_until(lambda: len(provider.inputs) == 2)
        main = h.session.registered_main
        assert main.decision.action_id == "L0001"
        h.scheduler.advance(29.0)
        await settle_until(lambda: provider.cancelled.is_set() and bool(h.platform.written))
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.execution_records[0].action_id == "L0000"
        assert final_review(h).status == "failed"
        assert not h.session.call_port.register_main(main)
        assert h.session.call_port.admit_call(binding, "repair").code == "DEADLINE"
        provider.release.set()
        await settle_until(lambda: len(h.session.call_results) == 2
                           and h.session.call_results[-1].usage.total_tokens == 17)
        late = h.session.call_results[-1]
        assert late.decision is None and late.error.code == "MODEL_CANCELLED"
        assert late.transport_requests == 0
        assert len(provider.inputs) == 2 and len(h.platform.written) == 1
        assert len(review_events(h)) == 1
        await h.session.shutdown()
        await gateway.aclose()
    asyncio.run(scenario())
