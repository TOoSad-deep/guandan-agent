"""B01 validates current DTOs, not the future model/review example as a whole."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from guandan_agent.contracts import (
    DecisionOutcome,
    DecisionProposal,
    EventDraft,
    ExecutionRecord,
    GateDecision,
    LegalActionSet,
    LogEvent,
    PlatformEvent,
    RequestBinding,
    RequestContext,
    RunSummary,
    SeatConnection,
    SubmissionIntent,
    WireAction,
)


@pytest.fixture
def context_data() -> dict:
    example = Path(__file__).resolve().parents[2] / "docs/examples/phase5-contract-example.json"
    return json.loads(example.read_text(encoding="utf-8"))["decision_input"]["context"]


@pytest.fixture
def context(context_data: dict) -> RequestContext:
    return RequestContext.model_validate(context_data)


def test_existing_example_context_round_trips_and_freezes(context_data: dict) -> None:
    context = RequestContext.model_validate(context_data)
    assert json.loads(context.model_dump_json()) == context_data
    assert RequestContext.model_validate_json(context.model_dump_json()) == context
    assert len(context.legal.actions) == 2  # Three raw entries include duplicate PASS.
    assert context.legal.actions[1].source_indices == (1, 2)
    context_data["state"]["self_hand"].append("SA")
    context_data["legal"]["actions"][0]["wire_action"][2].clear()
    assert context.state.self_hand == ("S9", "D4")
    assert context.legal.actions[0].wire_action == ("Single", "9", ("S9",))
    with pytest.raises(ValidationError, match="frozen"):
        context.state.stage = "back"
    with pytest.raises(TypeError):
        context.state.self_hand[0] = "SA"


@pytest.mark.parametrize("field,value", [("seat", "0"), ("seat", True), ("round_id", 0), ("run_id", "demo"), ("state_version", 1.0)])
def test_binding_rejects_coercion_and_invalid_identity(context_data: dict, field: str, value) -> None:
    binding = context_data["state"]["binding"]
    binding[field] = value
    with pytest.raises(ValidationError):
        RequestBinding.model_validate(binding)


def test_unknown_nested_field_and_cross_session_request_rejected(context_data: dict) -> None:
    context_data["state"]["other_hands"] = [["SA"]]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RequestContext.model_validate(context_data)
    binding = context_data["state"]["binding"]
    binding["request_id"] = "00000000-0000-4000-8000-000000000004:q7"
    with pytest.raises(ValidationError, match="request_id"):
        RequestBinding.model_validate(binding)


@pytest.mark.parametrize("action", [
    ["PASS", "PASS", []], ["PASS", "PASS", ["SA"]], ["Single", "PASS", ["SA"]],
    ["Single", "9", ["S10"]], ["Single", "9", []], ["Single", "9", {"S9"}],
])
def test_wire_action_rejects_malformed_pass_and_cards(action) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(WireAction).validate_python(action, strict=True)


def test_duplicate_physical_cards_and_order_are_preserved(context_data: dict) -> None:
    context_data["state"]["self_hand"] = ["S9", "S9", "D9"]
    context_data["state"]["remaining_counts"][0] = 3
    context_data["legal"]["actions"][0]["wire_action"] = ["Pair", "9", ["S9", "S9"]]
    loaded = RequestContext.model_validate(context_data)
    assert loaded.legal.actions[0].wire_action[2] == ("S9", "S9")
    context_data["state"]["self_hand"] = ["S9", "D9"]
    with pytest.raises(ValidationError, match="own hand"):
        RequestContext.model_validate(context_data)
    legal = deepcopy(context_data["legal"])
    legal["raw_count"] = 2
    legal["actions"] = [
        {"action_id": "L0000", "source_indices": [0], "wire_action": ["Pair", "9", ["S9", "D9"]]},
        {"action_id": "L0001", "source_indices": [1], "wire_action": ["Pair", "9", ["D9", "S9"]]},
    ]
    assert len(LegalActionSet.model_validate(legal).actions) == 2


@pytest.mark.parametrize("change", ["hole", "duplicate_index", "duplicate_action", "wrong_id", "empty"])
def test_legal_set_requires_complete_exact_origins(context_data: dict, change: str) -> None:
    legal = context_data["legal"]
    if change == "hole":
        legal["raw_count"] = 4
    elif change == "duplicate_index":
        legal["actions"][1]["source_indices"] = [0, 2]
    elif change == "duplicate_action":
        legal["actions"][1]["wire_action"] = legal["actions"][0]["wire_action"]
    elif change == "wrong_id":
        legal["actions"][1]["action_id"] = "L0002"
    else:
        legal["actions"] = []
        legal["raw_count"] = 0
    with pytest.raises(ValidationError):
        LegalActionSet.model_validate(legal)


@pytest.mark.parametrize("change", ["leader", "binding", "fallback", "history", "card_count_shape", "turn"])
def test_request_cross_field_consistency(context_data: dict, change: str) -> None:
    if change == "leader":
        context_data["state"]["leader_seat"] = None
    elif change == "binding":
        context_data["legal"]["binding"]["state_version"] += 1
    elif change == "fallback":
        context_data["fallback_action_id"] = "L9999"
    elif change == "history":
        context_data["state"]["history_complete"] = True
    elif change == "card_count_shape":
        context_data["state"]["remaining_counts"] = [2, 3]
    else:
        context_data["state"]["turn_seat"] = 1
    with pytest.raises(ValidationError):
        RequestContext.model_validate(context_data)


def test_back_context_is_frozen_and_required(context_data: dict) -> None:
    context_data["state"]["stage"] = "back"
    context_data["legal"]["raw_count"] = 1
    context_data["legal"]["actions"] = [{
        "action_id": "L0000", "source_indices": [0], "wire_action": ["back", "back", ["D4"]]
    }]
    with pytest.raises(ValidationError, match="platform_context"):
        RequestContext.model_validate(context_data)
    context_data["platform_context"] = {"tribute_pos": 2, "tribute_card": "SA"}
    context = RequestContext.model_validate(context_data)
    intent_data = {
        "submission_id": "submission-1", "binding": context.state.binding,
        "action_id": "L0000", "wire_action": context.legal.actions[0].wire_action,
        "stage": "back", "platform_context": context.platform_context, "final_deadline_mono": 30.0,
    }
    intent = SubmissionIntent.model_validate(intent_data)
    context_data["platform_context"]["tribute_card"] = "S2"
    assert intent.platform_context.tribute_card == "SA"
    intent_data["stage"] = "play"
    with pytest.raises(ValidationError):
        SubmissionIntent.model_validate(intent_data)


def test_rule_proposal_does_not_claim_future_model_support(context: RequestContext) -> None:
    proposal_data = {
        "binding": context.state.binding, "candidate_snapshot_id": None,
        "action_id": "L0000", "source": "rule", "reason": "fixed test",
        "cause_code": None, "review": None,
    }
    proposal = DecisionProposal.model_validate(proposal_data)
    assert DecisionOutcome(status="proposed", proposal=proposal, error=None).proposal == proposal
    with pytest.raises(ValidationError):
        DecisionOutcome(status="invalidated", proposal=proposal, error=None)
    with pytest.raises(ValidationError):
        DecisionOutcome(status="proposed", proposal=None, error=None)
    proposal_data["source"] = "main"
    with pytest.raises(ValidationError):
        DecisionProposal.model_validate(proposal_data)
    with pytest.raises(ValidationError):
        GateDecision(status="defer", error=None)


def draft_data(context: RequestContext) -> dict:
    binding = context.state.binding
    return {
        "schema_version": "1.0", "run_id": binding.run_id,
        "match_id": binding.match_id, "session_id": binding.session_id,
        "seat": binding.seat, "round_id": binding.round_id,
        "request_id": binding.request_id, "event_type": "request_opened",
        "payload": {"cards": ["S9", "S9"], "detail": {"usage": None}},
    }


def test_log_payload_deep_freeze_and_json_output(context: RequestContext) -> None:
    data = draft_data(context)
    event = LogEvent.model_validate({
        **data, "seq": 3, "event_id": f"{context.state.binding.run_id}:e3",
        "recorded_at": "2026-10-19T02:00:00Z",
    })
    data["payload"]["detail"]["usage"] = 999
    data["payload"]["cards"].clear()
    assert event.payload["detail"]["usage"] is None
    assert event.payload["cards"] == ("S9", "S9")
    with pytest.raises(TypeError):
        event.payload["detail"]["usage"] = 0
    dumped = json.loads(event.model_dump_json())
    assert dumped["payload"] == {"cards": ["S9", "S9"], "detail": {"usage": None}}
    assert LogEvent.model_validate_json(event.model_dump_json()) == event
    dumped["event_id"] = f"{context.state.binding.run_id}:e4"
    with pytest.raises(ValidationError):
        LogEvent.model_validate(dumped)


@pytest.mark.parametrize("payload", [{1: "bad"}, {"bad": float("nan")}, {"bad": object()}, {"bad": {"S9"}}])
def test_non_json_log_payloads_rejected(context: RequestContext, payload) -> None:
    data = draft_data(context)
    data["payload"] = payload
    with pytest.raises(ValidationError):
        EventDraft.model_validate(data)


def test_execution_confirmation_can_arrive_before_sent(context: RequestContext) -> None:
    record = ExecutionRecord(
        submission_id="submission-1", binding=context.state.binding,
        action_id="L0000", wire_action=context.legal.actions[0].wire_action,
        source="rule", status="confirmed", reserved_at="2026-10-19T02:00:00Z",
        write_started_at="2026-10-19T02:00:00.1Z", sent_at=None,
        confirmed_at="2026-10-19T02:00:00.2Z", feedback_event_id="observed-1", error=None,
    )
    assert record.sent_at is None
    summary = RunSummary(
        schema_version="1.0", run_id=context.state.binding.run_id,
        config_id="test-config", closed=True, log_complete=False,
        last_persisted_seq=0, pending_call_ids=[],
        unconfirmed_submission_ids=["submission-2"], error=None,
    )
    assert summary.unconfirmed_submission_ids == ("submission-2",)


def test_platform_envelope_and_connection(context: RequestContext) -> None:
    data = {
        "url": "ws://127.0.0.1:18181", "mode": "create", "room_id": None,
        "user_name": "test-seat-0", "seat": 0, "session_id": context.state.binding.session_id,
        "connection_epoch": 1, "server_rounds": 1,
    }
    assert SeatConnection.model_validate(data).mode == "create"
    data["mode"] = "join"
    with pytest.raises(ValidationError, match="room_id"):
        SeatConnection.model_validate(data)
    data["room_id"] = 1
    assert SeatConnection.model_validate(data).room_id == 1
    event = PlatformEvent(
        session_id=context.state.binding.session_id, connection_epoch=1, rx_seq=1,
        received_at=context.received_at, received_mono=10.0, kind="unknown", payload={"raw": []},
    )
    assert event.payload["raw"] == ()
