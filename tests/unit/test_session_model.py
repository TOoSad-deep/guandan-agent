"""Call authority, immutable candidates and late results on virtual time."""

import asyncio

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    ActionFeatures, CallTicket, Candidate, CandidateSnapshot, DecisionOutcome,
    DecisionProposal, ErrorInfo, MainReady, ModelCallResult, ModelDecision,
    ProposalReady, SeatConnection, ToolResult, ToolsReady, Usage,
)
from guandan_agent.runtime.budget import RunCallBudget
from guandan_agent.runtime.scheduling import ManualScheduler
from guandan_agent.runtime.session import Session
from guandan_agent.strategy import DecisionPolicy
from test_session import ControlledPlatform, Harness, MemoryRecorder, RUN, MATCH, begin, request, play

HAND = ("S3", "H9")
ACTIONS = [["Single", "3", ["S3"]], ["Single", "9", ["H9"]]]


def error(code="MODEL_SCHEMA"):
    return ErrorInfo(code=code, component="test", safe_message="Synthetic test failure", related_event_id=None)


def snapshot(context, *, ids=None, degraded=False):
    actions = context.legal.actions
    ids = ids if ids is not None else [a.action_id for a in actions]
    return CandidateSnapshot(
        binding=context.state.binding, candidate_snapshot_id=f"{context.state.binding.request_id}:c1",
        candidates=tuple(Candidate(
            action_id=identifier,
            features=ActionFeatures(played_count=1, remaining_count=1, estimated_turns=None,
                                    single_count=None, split_cost=None, is_bomb=False,
                                    finishes_hand=False, beats_leader=context.state.leader_seat is not None, rule_score=1.0),
            keep_reasons=("rule",),
        ) for identifier in ids), rule_action_id=ids[-1],
        tool_status="degraded" if degraded else "ok",
        degradation_codes=("ENGINE_UNAVAILABLE",) if degraded else (),
    )


class Tools:
    def __init__(self, *, fail=False, degraded=False, block=False):
        self.calls = []
        self.fail = fail
        self.degraded = degraded
        self.ready = asyncio.Event()
        if not block:
            self.ready.set()

    async def prepare(self, context, limits):
        self.calls.append((context, limits))
        await self.ready.wait()
        if self.fail:
            return ToolResult(snapshot=None, error=error("ENGINE_UNAVAILABLE"))
        return ToolResult(snapshot=snapshot(context, degraded=self.degraded),
                          error=error("ENGINE_UNAVAILABLE") if self.degraded else None)


class HoldingPolicy:
    def __init__(self):
        self.inputs = []
        self.calls = []
        self.ready = asyncio.Event()

    async def decide(self, input, calls=None):
        self.inputs.append(input)
        self.calls.append(calls)
        await self.ready.wait()
        return DecisionOutcome(status="unavailable", proposal=None, error=error())


def make(*, policy_name="V1", tools=None, policy=None, budget=None, seat=0):
    clock = ManualClock(100.0)
    scheduler = ManualScheduler(clock)
    recorder = MemoryRecorder()
    platform = ControlledPlatform()
    config = RunConfig.model_validate({"run": {"policy": policy_name}})
    connection = SeatConnection(
        url="ws://127.0.0.1:18181", mode="create", room_id=None, user_name=f"model-{seat}",
        seat=seat, session_id=f"00000000-0000-4000-8000-00000000006{seat}",
        connection_epoch=1, server_rounds=1,
    )
    session = Session(run_id=RUN, match_id=MATCH, connection=connection, config=config,
                      platform=platform, recorder=recorder, clock=clock, scheduler=scheduler,
                      tools=tools or Tools(), policy=policy or HoldingPolicy(), call_budget=budget)
    h = Harness(session, clock, scheduler, recorder, platform)
    h.post({"type": "CREATE_ROOM", "code": 200, "data": {"roomId": 1, "userNum": seat}})
    h.post({**begin(HAND), "myPos": seat})
    return h


def act(h):
    counts = [27, 27, 27, 27]
    counts[h.session.connection.seat] = 2
    h.post(request(hand=HAND, actions=ACTIONS, counts=counts))
    return h.session.active_context.state.binding


def result(ticket, *, decision=True, code="MODEL_SCHEMA", started=1, transport=0):
    return ModelCallResult(
        binding=ticket.binding, candidate_snapshot_id=f"{ticket.binding.request_id}:c1",
        role=ticket.role, call_id=ticket.call_id,
        decision=ModelDecision(action_id="L0001", reason="Preserve the lower card") if decision else None,
        error=None if decision else error(code), elapsed_ms=5.0, queue_wait_ms=0.0,
        calls_started=started, transport_requests=transport, usage=Usage(),
    )


def register(h, ticket, reply=None):
    reply = reply or result(ticket)
    h.session.call_port.record_result(reply)
    main = MainReady(binding=ticket.binding, candidate_snapshot_id=reply.candidate_snapshot_id,
                     call_id=ticket.call_id, decision=reply.decision)
    assert h.session.call_port.register_main(main)
    return main


def test_unique_skips_tools_and_model_even_with_v1():
    async def scenario():
        tools = Tools()
        h = make(tools=tools, policy=DecisionPolicy())
        h.post(request(hand=HAND, actions=[ACTIONS[0], ACTIONS[0]], counts=[2, 27, 27, 27]))
        await h.tick()
        assert not tools.calls
        assert h.session.execution_records[0].source == "unique"
        assert h.session.call_port.status(h.session.active_context.state.binding).slots_used == 0
        await h.session.shutdown()
    asyncio.run(scenario())


def test_valid_degraded_candidates_are_registered_once_and_only_narrow_port_leaves_session():
    async def scenario():
        policy = HoldingPolicy()
        h = make(tools=Tools(degraded=True), policy=policy)
        binding = act(h)
        await h.tick()
        registered = h.session.candidate_snapshot
        assert registered.tool_status == "degraded"
        assert policy.inputs[0].candidates == registered
        assert policy.calls[0] is h.session.call_port and policy.calls[0] is not h.session
        assert not hasattr(policy.calls[0], "post") and not hasattr(policy.calls[0], "platform")
        altered = registered.model_copy(update={"rule_action_id": "L0000"})
        h.session.post(ToolsReady(binding=binding, result=ToolResult(snapshot=altered, error=None)))
        assert h.session.candidate_snapshot == registered
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["missing", "unknown", "stale"])
def test_tools_fail_or_mismatched_candidates_fall_back_without_model(kind):
    async def scenario():
        h = make(tools=Tools(block=True))
        binding = act(h)
        valid = snapshot(h.session.active_context)
        if kind == "missing":
            tool_result = ToolResult(snapshot=None, error=error("TOOL_TIMEOUT"))
        elif kind == "unknown":
            tool_result = ToolResult(snapshot=snapshot(h.session.active_context, ids=["L9999"]), error=None)
        else:
            stale = binding.model_copy(update={"state_version": binding.state_version + 1})
            tool_result = ToolResult(snapshot=valid.model_copy(update={"binding": stale}), error=None)
        h.session.post(ToolsReady(binding=binding, result=tool_result))
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.candidate_snapshot is None
        assert h.session.call_port.status(binding).slots_used == 0
        await h.session.shutdown()
    asyncio.run(scenario())


def test_late_tools_after_stop_cannot_register_candidates_or_write():
    async def scenario():
        h = make(tools=Tools(block=True))
        binding = act(h)
        ready = ToolsReady(binding=binding, result=ToolResult(snapshot=snapshot(h.session.active_context), error=None))
        h.session.stop()
        h.session.post(ready)
        await h.tick()
        assert h.session.candidate_snapshot is None and not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_request_v0_v1_v2_limits_and_main_order():
    async def scenario():
        for policy, limit in (("V0", 0), ("V1", 1), ("V2", 2)):
            h = make(policy_name=policy)
            binding = act(h)
            await h.tick()
            calls = h.session.call_port
            assert calls.status(binding).max_slots == limit
            assert isinstance(calls.admit_call(binding, "repair"), ErrorInfo)
            ticket = calls.admit_call(binding, "main")
            if limit == 0:
                assert isinstance(ticket, ErrorInfo) and calls.status(binding).slots_used == 0
            else:
                assert isinstance(ticket, CallTicket)
                assert calls.begin_call(ticket).status == "allow"
                assert calls.begin_call(ticket).status == "deny"
                assert isinstance(calls.admit_call(binding, "main"), ErrorInfo)
                calls.record_result(result(ticket, decision=False))
                second = calls.admit_call(binding, "repair")
                if limit == 1:
                    assert isinstance(second, ErrorInfo)
                else:
                    assert isinstance(second, CallTicket) and second.ordinal == 2
                    assert calls.begin_call(second).status == "allow"
                    calls.record_result(result(second, decision=False))
                    assert isinstance(calls.admit_call(binding, "repair"), ErrorInfo)
            await h.session.shutdown()
    asyncio.run(scenario())


def test_t12_stop_between_admit_and_begin_denies_without_refunding_run_slot():
    async def scenario():
        budget = RunCallBudget(1)
        h = make(budget=budget)
        binding = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(binding, "main")
        h.session.stop()
        assert h.session.call_port.begin_call(ticket).status == "deny"
        assert budget.slots_used == 1 and budget.calls_started == 0
        h.session.call_port.record_result(result(ticket, decision=False, started=0))
        assert h.session.call_results[0].calls_started == 0
        await h.session.shutdown()
    asyncio.run(scenario())


def test_shared_run_budget_across_seats_prevents_second_seat_call():
    async def scenario():
        budget = RunCallBudget(1)
        left, right = make(budget=budget, seat=0), make(budget=budget, seat=1)
        first, second = act(left), act(right)
        await left.tick()
        await right.tick()
        ticket = left.session.call_port.admit_call(first, "main")
        assert isinstance(ticket, CallTicket)
        denied = right.session.call_port.admit_call(second, "main")
        assert isinstance(denied, ErrorInfo) and denied.code == "BATCH_BUDGET"
        assert budget.slots_used == 1
        assert right.session.call_port.begin_call(ticket).status == "deny"
        assert left.session.call_port.begin_call(ticket).status == "allow"
        await left.session.shutdown()
        await right.session.shutdown()
    asyncio.run(scenario())


def test_t12_relevant_fact_and_new_request_reject_old_call_and_late_main():
    async def scenario():
        h = make()
        old = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(old, "main")
        assert h.session.call_port.begin_call(ticket).status == "allow"
        other = ["Single", "4", ["S4"]]
        h.post(play(1, other))
        h.post(request(hand=HAND, actions=[ACTIONS[1]], counts=[2, 26, 27, 27],
                       context=(1, other), leader=(1, other)))
        current = h.session.active_context.state.binding
        late = result(ticket)
        h.session.call_port.record_result(late)
        assert not h.session.call_port.register_main(MainReady(
            binding=old, candidate_snapshot_id=late.candidate_snapshot_id,
            call_id=ticket.call_id, decision=late.decision))
        assert h.session.call_port.begin_call(ticket).status == "deny"
        assert h.session.active_context.state.binding == current
        assert h.session.call_port.status(current).slots_used == 0
        assert h.session.call_results == (late,)
        await h.session.shutdown()
    asyncio.run(scenario())


def test_main_registration_requires_finished_started_call_and_matching_decision():
    async def scenario():
        h = make()
        binding = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(binding, "main")
        reply = result(ticket)
        ready = MainReady(binding=binding, candidate_snapshot_id=reply.candidate_snapshot_id,
                          call_id=ticket.call_id, decision=reply.decision)
        assert not h.session.call_port.register_main(ready)
        h.session.call_port.record_result(reply)
        assert not h.session.call_results  # An unstarted call cannot claim success.
        assert h.session.call_port.begin_call(ticket).status == "allow"
        h.session.call_port.record_result(reply)
        changed = ready.model_copy(update={"decision": ModelDecision(action_id="L0000", reason="Different")})
        assert not h.session.call_port.register_main(changed)
        assert h.session.call_port.register_main(ready)
        assert not h.session.execution_records  # Registration is not submission.
        assert h.session.call_port.register_main(ready)
        assert len([e for e in h.recorder.events if e.event_type == "main_registered"]) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t04_deadline_keeps_valid_registered_main_instead_of_cheap_fallback():
    async def scenario():
        h = make()
        binding = act(h)
        await h.tick()
        assert h.session.active_context.fallback_action_id == "L0000"
        ticket = h.session.call_port.admit_call(binding, "main")
        assert h.session.call_port.begin_call(ticket).status == "allow"
        main = register(h, ticket)
        h.scheduler.advance(29.0)
        await h.tick()
        record = h.session.execution_records[0]
        assert record.source == "main" and record.action_id == main.decision.action_id == "L0001"
        assert len(h.platform.written) == 1
        assert not h.session.call_port.status(binding).suggestions_open
        assert not h.session.call_port.register_main(main)
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_deadline_without_registered_main_falls_back_and_late_result_cannot_replace():
    async def scenario():
        h = make()
        binding = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(binding, "main")
        assert h.session.call_port.begin_call(ticket).status == "allow"
        h.scheduler.advance(29.0)
        await h.tick()
        reply = result(ticket)
        h.session.call_port.record_result(reply)
        assert not h.session.call_port.register_main(MainReady(
            binding=binding, candidate_snapshot_id=reply.candidate_snapshot_id,
            call_id=ticket.call_id, decision=reply.decision))
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.execution_records[0].action_id == "L0000"
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_started_second_call_only_switches_watchdog_to_fallback():
    async def scenario():
        for start_second in (False, True):
            h = make(policy_name="V2")
            binding = act(h)
            await h.tick()
            first = h.session.call_port.admit_call(binding, "main")
            assert h.session.call_port.begin_call(first).status == "allow"
            register(h, first)
            second = h.session.call_port.admit_call(binding, "review")
            assert isinstance(second, CallTicket)
            if start_second:
                assert h.session.call_port.begin_call(second).status == "allow"
            h.scheduler.advance(29.0)
            await h.tick()
            assert h.session.execution_records[0].source == ("fallback" if start_second else "main")
            await h.session.shutdown()
    asyncio.run(scenario())


def test_t13_old_usage_is_evidence_only_and_closed_session_ignores_callbacks():
    async def scenario():
        h = make()
        old = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(old, "main")
        h.session.call_port.begin_call(ticket)
        other = ["Single", "4", ["S4"]]
        h.post(play(1, other))
        h.post(request(hand=HAND, actions=ACTIONS, counts=[2, 26, 27, 27],
                       context=(1, other), leader=(1, other)))
        new = h.session.active_context.state.binding
        await h.tick()
        second = h.session.call_port.admit_call(new, "main")
        reply = result(ticket, decision=False, code="MODEL_TIMEOUT", transport=None)
        h.session.call_port.record_result(reply)
        assert h.session.call_port.begin_call(second).status == "allow"
        assert h.session.call_port.status(new).slots_used == 1
        assert h.session.registered_main is None
        assert h.session.call_results[0].usage.total_tokens is None
        assert h.session.call_results[0].transport_requests is None
        assert h.session.pending_call_ids == (second.call_id,)
        await h.session.shutdown()
        before = len(h.recorder.events)
        h.session.call_port.record_result(result(second))
        assert len(h.recorder.events) == before
        assert h.session.call_results == (reply,)
        assert all("deadline_mono" not in str(e.payload) for e in h.recorder.events)
    asyncio.run(scenario())


def test_fake_model_proposal_without_registered_main_is_rejected():
    async def scenario():
        h = make()
        binding = act(h)
        await h.tick()
        h.session.post(ProposalReady(binding=binding, outcome=DecisionOutcome(
            status="proposed", error=None,
            proposal=DecisionProposal(binding=binding, candidate_snapshot_id=f"{binding.request_id}:c1",
                                      action_id="L0001", source="main", reason="Forged", cause_code=None, review=None))))
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert h.session.execution_records[0].action_id == "L0000"
        await h.session.shutdown()
    asyncio.run(scenario())


def test_late_telemetry_fills_unknowns_but_never_changes_failed_decision():
    async def scenario():
        h = make()
        binding = act(h)
        await h.tick()
        ticket = h.session.call_port.admit_call(binding, "main")
        assert h.session.call_port.begin_call(ticket).status == "allow"
        timeout = result(ticket, decision=False, code="MODEL_TIMEOUT", transport=None)
        h.session.call_port.record_result(timeout)
        late = result(ticket).model_copy(update={"transport_requests": 1,
                                                "usage": Usage(input_tokens=5, output_tokens=2, total_tokens=7)})
        h.session.call_port.record_result(late)
        updated = h.session.call_results[0]
        assert updated.decision is None and updated.error.code == "MODEL_TIMEOUT"
        assert updated.usage.total_tokens == 7 and updated.transport_requests == 1
        assert not h.session.call_port.register_main(MainReady(
            binding=binding, candidate_snapshot_id=late.candidate_snapshot_id,
            call_id=ticket.call_id, decision=late.decision))
        count = len(h.recorder.events)
        h.session.call_port.record_result(late)
        assert len(h.recorder.events) == count
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("response,source", [
    ('{"action_id":"L0001","reason":"Keep the low card"}', "main"),
    ('{"action_id":"L9999","reason":"Invalid member"}', "fallback"),
    ('not-json', "fallback"),
])
def test_t02_real_v1_graph_with_scripted_provider_and_session_authority(response, source):
    from guandan_agent.model import ModelGateway, ScriptedProvider

    async def scenario():
        h = make()
        provider = ScriptedProvider([response], clock=h.clock)
        gateway = ModelGateway(provider, clock=h.clock)
        h.session.policy = DecisionPolicy(gateway=gateway)
        binding = act(h)
        for _ in range(100):
            await asyncio.sleep(0)
            if h.platform.written:
                break
        assert len(provider.inputs) == 1
        assert h.session.call_port.status(binding).slots_used == 1
        record = h.session.execution_records[0]
        assert record.source == source
        expected = ACTIONS[1] if source == "main" else ACTIONS[0]
        h.post(play(0, expected))
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1
        assert h.session.call_results[0].calls_started == 1
        assert h.session.call_results[0].transport_requests == 0
        await h.session.shutdown()
        assert not provider.closed  # Session never closes resources shared by a run.
        await provider.aclose()
    asyncio.run(scenario())
