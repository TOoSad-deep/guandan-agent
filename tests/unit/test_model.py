"""Offline transport and gateway boundaries; no real credentials or requests."""

import asyncio
import json

import httpx
import pytest

from guandan_agent.clock import ManualClock, SystemClock
from guandan_agent.contracts import (
    ActionFeatures, BudgetView, CallTicket, Candidate, CandidateSnapshot,
    GateDecision, ProviderInput, ProviderMessage, PublicPlay, Usage,
)
from guandan_agent.model import DeepSeekProvider, ModelGateway, ScriptedProvider, ScriptedStep, build_model_input
from guandan_agent.model.gateway import build_messages
from guandan_agent.model.provider import normalize_usage, safe_error
from tests.unit.test_rules import decision_input, make_state_and_legal, PASS


def model_case(*, policy="V1", candidates=None, history=()):
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS])
    state = state.model_copy(update={"history": history, "history_complete": True})
    input = decision_input(state, legal, policy=policy, fallback="L0001")
    features = ActionFeatures(played_count=1, remaining_count=1, estimated_turns=None,
                              single_count=None, split_cost=None, is_bomb=False,
                              finishes_hand=False, beats_leader=True, rule_score=1.0)
    snapshot = CandidateSnapshot(
        binding=state.binding, candidate_snapshot_id=f"{state.binding.request_id}:c1",
        candidates=tuple(Candidate(action_id=name, features=features, keep_reasons=("ranked",))
                         for name in (candidates or ("L0000", "L0001"))),
        rule_action_id="L0000", tool_status="ok", degradation_codes=(),
    )
    return input.model_copy(update={"candidates": snapshot})


class FakeCalls:
    """A narrow authority fake; tests assert effects through this public surface."""
    def __init__(self, binding, *, clock=None, deadline=100.0):
        self.binding = binding
        self.clock = clock or ManualClock()
        self.deadline = deadline
        self.active = True
        self.slots = 0
        self.started = 0
        self.results = []
        self.registered = []
        self.before_begin = None
        self.allow_registration = True

    def status(self, binding):
        remaining = max(0.0, self.deadline - self.clock.monotonic())
        return BudgetView(active=self.active and binding == self.binding,
                          suggestions_open=self.active and remaining > 0,
                          remaining_s=remaining, slots_used=self.slots, max_slots=1)

    def admit_call(self, binding, role):
        if not self.status(binding).active or not self.status(binding).suggestions_open:
            return safe_error("REQUEST_STALE", "Stale request.")
        if self.slots >= 1:
            return safe_error("CALL_BUDGET", "Call budget exhausted.")
        self.slots += 1
        return CallTicket(call_id="test-call-1", binding=binding, role=role,
                          ordinal=1, deadline_mono=self.deadline)

    def begin_call(self, ticket):
        if self.before_begin:
            self.before_begin()
        if not self.status(ticket.binding).active:
            return GateDecision(status="deny", error=safe_error("REQUEST_STALE", "Stale request."))
        self.started += 1
        return GateDecision(status="allow", error=None)

    def record_result(self, result):
        self.results.append(result)

    def register_main(self, main):
        assert self.results and self.results[-1].decision == main.decision
        if self.allow_registration:
            self.registered.append(main)
            return True
        return False


def provider_input(clock, *, deadline=None):
    return ProviderInput(call_id="offline-report-1", messages=(
        ProviderMessage(role="system", content="Return a json report."),
        ProviderMessage(role="user", content="Summarize the already closed record."),
    ), deadline_mono=clock.monotonic() + 1 if deadline is None else deadline, max_output_tokens=123)


def completion(content='{"summary":"offline report"}', usage=None):
    return {"id": "mock-completion", "object": "chat.completion", "created": 1,
            "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}],
            **({"usage": usage} if usage is not None else {})}


def test_real_adapter_json_mode_limit_thinking_and_offline_contract():
    async def scenario():
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=completion(usage={"prompt_tokens": 12, "completion_tokens": 4,
                                                             "total_tokens": 16}))
        clock = SystemClock()
        provider = DeepSeekProvider(api_key="fake-unit-test-key", clock=clock,
                                    transport=httpx.MockTransport(handle))
        try:
            reply = await provider.complete(provider_input(clock))
        finally:
            await provider.aclose()
        assert reply.error is None and reply.content == '{"summary":"offline report"}'
        assert reply.transport_requests == 1 and reply.usage.total_tokens == 16
        assert len(requests) == 1
        assert requests[0]["response_format"] == {"type": "json_object"}
        assert requests[0]["max_tokens"] == 123
        assert requests[0]["thinking"] == {"type": "disabled"}
        assert "binding" not in str(requests[0]) and "candidates" not in str(requests[0])
    asyncio.run(scenario())


def test_transport_500_has_no_retry_and_no_provider_body_leak():
    async def scenario():
        count = 0
        def handle(request):
            nonlocal count
            count += 1
            return httpx.Response(500, json={"error": {"message": "PRIVATE-provider-diagnostic"}})
        provider = DeepSeekProvider(api_key="fake-key", transport=httpx.MockTransport(handle))
        try:
            reply = await provider.complete(provider_input(SystemClock()))
        finally:
            await provider.aclose()
        assert count == reply.transport_requests == 1
        assert reply.error.code == "MODEL_TRANSPORT"
        assert "PRIVATE" not in reply.model_dump_json()
        assert reply.usage.total_tokens is None and reply.usage.cost_amount is None
    asyncio.run(scenario())


def test_provider_empty_content_is_raw_and_not_an_offline_schema_failure():
    async def scenario():
        provider = DeepSeekProvider(api_key="fake-key", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion(content=""))))
        try:
            reply = await provider.complete(provider_input(SystemClock()))
        finally:
            await provider.aclose()
        assert reply.content == "" and reply.error is None
        assert reply.usage == Usage()
    asyncio.run(scenario())


def test_provider_timeout_is_bounded_and_cancel_is_not_success():
    async def scenario():
        entered = asyncio.Event()
        async def handle(request):
            entered.set()
            await asyncio.Event().wait()
        provider = DeepSeekProvider(api_key="fake-key", transport=httpx.MockTransport(handle))
        clock = SystemClock()
        try:
            timed = await provider.complete(provider_input(clock, deadline=clock.monotonic() + 0.02))
            assert timed.error.code == "MODEL_TIMEOUT" and timed.transport_requests == 1
            entered.clear()
            task = asyncio.create_task(provider.complete(provider_input(clock)))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await provider.aclose()
    asyncio.run(scenario())


def test_missing_key_is_rejected_even_if_environment_contains_a_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-loaded")
    with pytest.raises(TypeError):
        DeepSeekProvider()
    with pytest.raises(ValueError, match="explicit"):
        DeepSeekProvider(api_key="")


def test_partial_usage_never_infers_total_or_price():
    usage = normalize_usage({"prompt_tokens": 12, "completion_tokens": 4})
    assert usage.input_tokens == 12 and usage.output_tokens == 4
    assert usage.total_tokens is None and usage.cost_amount is None
    assert normalize_usage({"prompt_tokens": True, "completion_tokens": -1}) == Usage()


def test_visible_projection_uses_original_candidates_and_no_other_private_fields():
    original = model_case()
    model_input = build_model_input(original)
    assert model_input.candidates[1].cards == ()
    assert model_input.candidates[0].cards == ("S9",)
    messages, size, omitted = build_messages(model_input, max_input_bytes=24576)
    baseline = messages
    polluted = model_input.visible.model_copy(update={"other_hands": [["HA"]], "api_key": "PRIVATE"})
    mutated = model_input.model_copy(update={"visible": polluted})
    assert build_messages(mutated, max_input_bytes=24576)[0] == baseline
    payload = json.loads(messages[1].content)
    assert "binding" not in payload["visible"]
    assert payload["visible"]["self_hand"] == ["S9", "D4"]
    assert "legal" not in payload and "source_indices" not in messages[1].content
    assert size > 0 and omitted == 0


def test_prompt_only_trims_old_history_and_marks_omission():
    history = tuple(PublicPlay(event_id=f"event-{index}-" + "x" * 30, seat=1,
                               action=("Single", "4", ("S4",))) for index in range(5))
    full = build_model_input(model_case(history=history))
    _, original_size, _ = build_messages(full, max_input_bytes=100000)
    messages, size, omitted = build_messages(full, max_input_bytes=original_size - 100)
    payload = json.loads(messages[1].content)
    assert omitted > 0 and size <= original_size - 100
    assert payload["visible"]["history"][0]["event_id"] == history[omitted].event_id
    assert payload["visible"]["history_complete"] is False
    assert len(payload["candidates"]) == 2 and payload["visible"]["self_hand"] == ["S9", "D4"]
    assert full.visible.history == history


@pytest.mark.parametrize("content,error", [
    (None, "MODEL_EMPTY"), ("", "MODEL_EMPTY"), (" ", "MODEL_SCHEMA"),
    ('{"action_id":"L0000","reason":" "}', "MODEL_SCHEMA"),
    ('{"action_id":"L0000","reason":"ok","cards":["S9"]}', "MODEL_SCHEMA"),
    ('{"action_id":0,"reason":"ok"}', "MODEL_SCHEMA"),
    ('{"action_id":"unknown","reason":"ok"}', "MODEL_ACTION_UNKNOWN"),
])
def test_gateway_strict_output_and_no_repair(content, error):
    async def scenario():
        input = build_model_input(model_case())
        port = FakeCalls(input.binding)
        provider = ScriptedProvider([content, '{"action_id":"L0000","reason":"unused"}'], clock=port.clock)
        reply = await ModelGateway(provider, clock=port.clock).choose(input, port)
        assert reply.error.code == error and reply.decision is None
        assert len(provider.inputs) == port.slots == port.started == 1
        assert reply.transport_requests == 0 and reply.usage.total_tokens is None
        assert port.results == [reply]
    asyncio.run(scenario())


def test_input_limit_takes_no_ticket_or_transport():
    async def scenario():
        input = build_model_input(model_case())
        port = FakeCalls(input.binding)
        provider = ScriptedProvider([])
        result = await ModelGateway(provider, clock=port.clock, max_input_bytes=1).choose(input, port)
        assert result.error.code == "INPUT_LIMIT" and result.call_id is None
        assert result.calls_started == result.transport_requests == port.slots == 0
        assert provider.inputs == []
    asyncio.run(scenario())


def test_admitted_ticket_rechecks_stop_before_provider():
    async def scenario():
        input = build_model_input(model_case())
        port = FakeCalls(input.binding)
        port.before_begin = lambda: setattr(port, "active", False)
        provider = ScriptedProvider([])
        result = await ModelGateway(provider, clock=port.clock).choose(input, port)
        assert port.slots == 1 and port.started == result.calls_started == 0
        assert result.call_id is not None and result.error.code == "REQUEST_STALE"
        assert provider.inputs == [] and result.transport_requests == 0
    asyncio.run(scenario())


def test_gateway_timeout_and_cancellation_count_started_with_unknown_remote_usage():
    async def scenario():
        input = build_model_input(model_case())
        clock = SystemClock()
        port = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 10)
        provider = ScriptedProvider([ScriptedStep(delay_s=10)])
        result = await ModelGateway(provider, clock=clock, timeout_s=0.01).choose(input, port)
        assert result.error.code == "MODEL_TIMEOUT" and result.calls_started == 1
        assert result.transport_requests in (None, 0) and result.usage == Usage()
        port = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 10)
        provider = ScriptedProvider([ScriptedStep(delay_s=10)])
        task = asyncio.create_task(ModelGateway(provider, clock=clock).choose(input, port))
        while not provider.inputs:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert port.results[-1].error.code == "MODEL_CANCELLED"
        assert port.results[-1].calls_started == 1 and port.results[-1].usage == Usage()
    asyncio.run(scenario())


def test_scripted_provider_enforces_its_own_offline_deadline_and_preserves_expired_step():
    async def scenario():
        clock = SystemClock()
        provider = ScriptedProvider([ScriptedStep(content='{"report":"ok"}', delay_s=0.04)], clock=clock)
        expired = await provider.complete(provider_input(clock, deadline=clock.monotonic() - 1))
        assert expired.error.code == "MODEL_TIMEOUT" and provider.inputs == []
        timed = await provider.complete(provider_input(clock, deadline=clock.monotonic() + 0.005))
        assert timed.error.code == "MODEL_TIMEOUT" and timed.content is None
        assert timed.transport_requests == 0 and len(provider.inputs) == 1
        await provider.aclose()
    asyncio.run(scenario())


def test_cancel_resistant_provider_cannot_block_gateway_or_revive_late_decision():
    async def scenario():
        from guandan_agent.contracts import ProviderReply
        release = asyncio.Event()
        class ResistantProvider:
            async def complete(self, input):
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return ProviderReply(call_id=input.call_id, content='{"action_id":"L0000","reason":"late"}',
                                     usage=Usage(total_tokens=9), transport_requests=1, elapsed_ms=50.0, error=None)
        clock = SystemClock()
        input = build_model_input(model_case())
        calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 5)
        gateway = ModelGateway(ResistantProvider(), clock=clock, timeout_s=0.01)
        result = await asyncio.wait_for(gateway.choose(input, calls), timeout=0.3)
        assert result.error.code == "MODEL_TIMEOUT" and result.decision is None
        assert result.transport_requests is None and result.usage.total_tokens is None
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls.results[-1].usage.total_tokens == 9
        assert calls.results[-1].error.code == "MODEL_TIMEOUT" and calls.results[-1].decision is None
        assert calls.slots == calls.started == 1
    asyncio.run(scenario())


def test_shared_concurrency_queue_rechecks_validity_without_spending_a_ticket():
    async def scenario():
        input = build_model_input(model_case())
        clock = SystemClock()
        ports = [FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 3) for _ in range(2)]
        provider = ScriptedProvider([ScriptedStep(content='{"action_id":"L0000","reason":"ok"}', delay_s=0.02)])
        gateway = ModelGateway(provider, clock=clock, concurrency=1)
        first = asyncio.create_task(gateway.choose(input, ports[0]))
        while not provider.inputs:
            await asyncio.sleep(0)
        second = asyncio.create_task(gateway.choose(input, ports[1]))
        await asyncio.sleep(0)
        ports[1].active = False
        results = await asyncio.gather(first, second)
        assert results[0].decision.action_id == "L0000"
        assert results[1].error.code == "REQUEST_STALE" and results[1].queue_wait_ms > 0
        assert ports[1].slots == 0 and len(provider.inputs) == 1
    asyncio.run(scenario())


def test_queue_deadline_records_wait_without_starting_a_call():
    async def scenario():
        input = build_model_input(model_case())
        clock = SystemClock()
        provider = ScriptedProvider([ScriptedStep(content='{"action_id":"L0000","reason":"ok"}', delay_s=0.05)])
        gateway = ModelGateway(provider, clock=clock)
        first_calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 1)
        first = asyncio.create_task(gateway.choose(input, first_calls))
        while not provider.inputs:
            await asyncio.sleep(0)
        queued_calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 0.01)
        result = await gateway.choose(input, queued_calls)
        assert result.error.code == "DEADLINE" and result.queue_wait_ms > 0
        assert queued_calls.slots == result.calls_started == result.transport_requests == 0
        assert result.call_id is None and len(provider.inputs) == 1
        await first
    asyncio.run(scenario())


def test_cancel_resistant_provider_keeps_concurrency_slot_until_it_finishes():
    async def scenario():
        from guandan_agent.contracts import ProviderReply
        release = asyncio.Event()
        class ResistantProvider:
            def __init__(self):
                self.calls = 0
                self.active = 0
                self.max_active = 0

            async def complete(self, input):
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        await release.wait()
                    return ProviderReply(call_id=input.call_id, content=None, usage=Usage(),
                                         transport_requests=1, elapsed_ms=50.0, error=None)
                finally:
                    self.active -= 1

        clock = SystemClock()
        input = build_model_input(model_case())
        provider = ResistantProvider()
        gateway = ModelGateway(provider, clock=clock, concurrency=1, timeout_s=0.01)
        first_calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 1)
        first = await asyncio.wait_for(gateway.choose(input, first_calls), timeout=0.3)
        assert first.error.code == "MODEL_TIMEOUT" and provider.active == 1
        second_calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 0.02)
        second = await asyncio.wait_for(gateway.choose(input, second_calls), timeout=0.3)
        assert second.error.code == "DEADLINE" and second.queue_wait_ms > 0
        assert second_calls.slots == second.calls_started == second.transport_requests == 0
        assert provider.calls == provider.max_active == provider.active == 1
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert provider.active == 0
        third_calls = FakeCalls(input.binding, clock=clock, deadline=clock.monotonic() + 1)
        await gateway.choose(input, third_calls)
        assert provider.calls == 2 and provider.max_active == 1
    asyncio.run(scenario())
