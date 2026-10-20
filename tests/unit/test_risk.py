"""Risk routing uses public, relative-seat facts and canonical action types."""

import pytest

from guandan_agent.config import ReviewConfig
from guandan_agent.contracts import ActionFeatures, Candidate, CandidateSnapshot, ModelDecision
from guandan_agent.strategy.risk import detect_risks
from tests.unit.test_rules import PASS, decision_input, make_state_and_legal


def risk_case(*, kind="Single", counts=None, leader=1, seat=0, finished=(), stage="play"):
    action = PASS if kind == "PASS" else (kind, "9", ("S9",))
    other = ("Single", "4", ("D4",))
    if stage != "play":
        action, other = (stage, stage, ("S9",)), (stage, stage, ("D4",))
    state, legal = make_state_and_legal(["S9", "D4"], [action, other],
                                      leader=leader, counts=counts, seat=seat, stage=stage)
    state = state.model_copy(update={"finished_order": finished})
    features = ActionFeatures(played_count=1, remaining_count=1, estimated_turns=1,
                              single_count=1, split_cost=3.0, is_bomb=False,
                              finishes_hand=False, beats_leader=False, rule_score=1.0)
    input = decision_input(state, legal, policy="V2")
    snapshot = CandidateSnapshot(
        binding=state.binding, candidate_snapshot_id=f"{state.binding.request_id}:c1",
        candidates=tuple(Candidate(action_id=action.action_id, features=features, keep_reasons=())
                         for action in legal.actions),
        rule_action_id="L0000", tool_status="ok", degradation_codes=(),
    )
    return input.model_copy(update={"candidates": snapshot})


def risks(input, action_id="L0000"):
    return detect_risks(input, ModelDecision(action_id=action_id, reason="choice"), ReviewConfig())


@pytest.mark.parametrize("count", [1, 2])
def test_any_unfinished_opponent_close_not_only_leader(count):
    assert risks(risk_case(counts=(2, None, 2, count), leader=1)) == ("OPPONENT_NEAR_FINISH",)


@pytest.mark.parametrize("counts,finished", [
    ((2, None, 1, None), ()), ((2, 0, 2, 0), ()), ((2, 1, 2, 2), (1, 3)),
    ((2, 3, 2, 4), ()),
])
def test_unknown_zero_partner_counts_and_finished_opponents_are_not_pressure(counts, finished):
    assert risks(risk_case(counts=counts, finished=finished)) == ()


@pytest.mark.parametrize("kind", ["Bomb", "StraightFlush", "FourKings"])
def test_bomb_risk_resolves_original_canonical_type_not_feature_flag(kind):
    assert risks(risk_case(kind=kind)) == ("USE_BOMB",)


def test_noncanonical_type_does_not_create_bomb_risk():
    assert risks(risk_case(kind="bomb")) == ()


@pytest.mark.parametrize("seat", range(4))
def test_partner_relation_is_relative_and_pass_never_beats_partner(seat):
    assert risks(risk_case(seat=seat, leader=(seat + 2) % 4)) == ("BEAT_PARTNER",)
    assert risks(risk_case(kind="PASS", seat=seat, leader=(seat + 2) % 4)) == ()


def test_multiple_risks_merge_once_in_stable_order():
    input = risk_case(kind="Bomb", leader=2, counts=(2, 1, 4, 2))
    assert risks(input) == ("OPPONENT_NEAR_FINISH", "USE_BOMB", "BEAT_PARTNER")


@pytest.mark.parametrize("stage", ["tribute", "back"])
def test_exchange_stage_does_not_apply_play_risks(stage):
    assert risks(risk_case(kind="Bomb", leader=2, counts=(2, 1, 4, 2), stage=stage)) == ()


def test_unmapped_or_non_candidate_decision_has_no_risk_and_split_cost_is_disabled():
    input = risk_case()
    assert risks(input, "missing") == ()
    snapshot = input.candidates.model_copy(update={"candidates": input.candidates.candidates[1:],
                                                   "rule_action_id": "L0001"})
    assert risks(input.model_copy(update={"candidates": snapshot})) == ()
    assert risks(input) == ()  # split_cost=3 is not an activated, calibrated threshold.
