"""B02 deterministic rules; no model, platform network or Java engine."""

import asyncio
from collections import Counter

import pytest

from guandan_agent.contracts import (
    DecisionInput,
    LegalActionSet,
    RequestBinding,
    RequestContext,
    VisibleState,
)
from guandan_agent.hand import CHEAP_RULE_VERSION, cheap_choice
from guandan_agent.strategy import DecisionPolicy


PASS = ("PASS", "PASS", "PASS")


def make_state_and_legal(
    hand, raw_actions, *, stage="play", level="2", leader=1, counts=None, seat=0,
):
    binding = RequestBinding(
        run_id="00000000-0000-4000-8000-000000000001",
        match_id="00000000-0000-4000-8000-000000000002",
        round_id=1,
        session_id="00000000-0000-4000-8000-000000000003",
        seat=seat, connection_epoch=1,
        request_id="00000000-0000-4000-8000-000000000003:q1", state_version=1,
    )
    action_data = {}
    for index, action in enumerate(raw_actions):
        key = (action[0], action[1], tuple(action[2]) if action[0] != "PASS" else "PASS")
        if key not in action_data:
            action_data[key] = {"action_id": f"L{index:04d}", "source_indices": [], "wire_action": key}
        action_data[key]["source_indices"].append(index)
    legal = LegalActionSet(binding=binding, raw_count=len(raw_actions), actions=list(action_data.values()))
    if counts is None:
        counts = [None] * 4
        counts[seat] = len(hand)
    state = VisibleState(
        binding=binding, stage=stage, self_hand=hand, level_rank=level,
        remaining_counts=counts, team_ranks=[None, None], turn_seat=seat,
        leader_seat=leader,
        leading_action=None if leader is None else ("Single", "3", ("C3",)),
        finished_order=[], history=[], history_complete=False,
        history_omitted=0, facts_status="trusted",
    )
    return state, legal


def decision_input(state, legal, *, fallback=None, policy="V0"):
    context = RequestContext(
        state=state, legal=legal, received_at="2026-10-19T02:00:00Z",
        fallback_action_id=fallback or legal.actions[0].action_id,
        policy=policy,
        platform_context={"tribute_pos": 2, "tribute_card": "SA"} if state.stage == "back" else None,
    )
    return DecisionInput(context=context)


class UnusableCalls:
    def __getattribute__(self, name):
        raise AssertionError("B02 must never access a model call port")


@pytest.mark.parametrize("policy", ["V0", "V1", "V2"])
def test_unique_uses_complete_deduplicated_set_without_rule_or_call_port(policy):
    pair = ("Pair", "9", ("S9", "S9"))
    state, legal = make_state_and_legal(["S9", "S9", "D4"], [pair, pair])

    def must_not_select(*_):
        raise AssertionError("unique shortcut must not run ranking")

    result = asyncio.run(DecisionPolicy(must_not_select).decide(
        decision_input(state, legal, policy=policy), UnusableCalls(),
    ))
    assert legal.raw_count == 2
    assert legal.actions[0].source_indices == (0, 1)
    assert result.proposal.source == "unique"
    assert result.proposal.action_id == "L0000"
    assert result.error is None


def test_duplicate_pass_does_not_make_a_multi_action_set_unique():
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS, PASS])
    seen = []

    def choose_pass(current_state, full_set):
        seen.append((current_state, full_set))
        return "L0001"

    result = asyncio.run(DecisionPolicy(choose_pass).decide(decision_input(state, legal), UnusableCalls()))
    assert len(seen) == 1
    assert seen[0][1] is legal
    assert result.proposal.source == "rule"
    assert result.proposal.action_id == "L0001"
    assert result.proposal.cause_code is None


def test_finishing_hand_takes_priority_over_partner_pass_and_bomb_conservation():
    cards = ("S9", "S9", "D9", "C9")
    state, legal = make_state_and_legal(cards, [PASS, ("Bomb", "9", cards)], leader=2)
    assert cheap_choice(state, legal) == "L0001"
    assert Counter(state.self_hand) == Counter(cards)


def test_first_lead_chooses_an_existing_non_pass_and_keeps_stable_origin_tie():
    state, legal = make_state_and_legal(
        ["D9", "S9", "H2"], [("Single", "9", ("D9",)), ("Single", "9", ("S9",))],
        leader=None,
    )
    for _ in range(3):
        assert cheap_choice(state, legal) == "L0000"
    assert all(action.wire_action != PASS for action in legal.actions)


def test_lead_does_not_choose_pass_when_a_platform_non_pass_exists():
    state, legal = make_state_and_legal(
        ["S9", "D9", "C9", "H9", "S3"],
        [PASS, ("Bomb", "9", ("S9", "D9", "C9", "H9"))], leader=None,
    )
    assert cheap_choice(state, legal) == "L0001"


@pytest.mark.parametrize("seat,partner", [(0, 2), (1, 3), (2, 0), (3, 1)])
def test_partner_leading_pass_uses_relative_seat_and_unknown_counts(seat, partner):
    state, legal = make_state_and_legal(
        ["S9", "D4"], [("Single", "9", ("S9",)), PASS],
        leader=partner, seat=seat, counts=[None, None, None, None],
    )
    assert cheap_choice(state, legal) == "L0001"


def test_normal_play_prefers_more_cards_and_preserves_a_bomb():
    state, legal = make_state_and_legal(
        ["S9", "D9", "C9", "H9", "S3", "D3", "S4"],
        [("Bomb", "9", ("S9", "D9", "C9", "H9")),
         ("Single", "4", ("S4",)), ("Pair", "3", ("S3", "D3")), PASS],
        leader=None,
    )
    assert cheap_choice(state, legal) == "L0002"


@pytest.mark.parametrize("opponent_count,expected", [(None, "L0001"), (0, "L0001"), (1, "L0000"), (2, "L0000"), (3, "L0001")])
def test_only_known_opponent_small_hand_justifies_spending_bomb(opponent_count, expected):
    state, legal = make_state_and_legal(
        ["S9", "D9", "C9", "H9", "S3"],
        [("Bomb", "9", ("S9", "D9", "C9", "H9")), PASS],
        counts=[5, opponent_count, 1, None],
    )
    assert cheap_choice(state, legal) == expected


def test_known_pressure_uses_opponent_parity_for_odd_seat():
    state, legal = make_state_and_legal(
        ["S9", "D9", "C9", "H9", "S3"],
        [("Bomb", "9", ("S9", "D9", "C9", "H9")), PASS],
        counts=[1, 5, None, None], seat=1, leader=0,
    )
    assert cheap_choice(state, legal) == "L0000"


def test_all_bomb_lead_preserves_four_kings_and_spends_fewer_other_cards():
    flush = ("S4", "S5", "S6", "S7", "S8")
    bomb = ("S9", "S9", "D9", "D9", "C9", "H9")
    kings = ("SB", "SB", "HR", "HR")
    state, legal = make_state_and_legal(
        [*kings, *bomb, *flush],
        [("FourKings", "B", kings), ("Bomb", "9", bomb), ("StraightFlush", "4", flush)],
        leader=None,
    )
    assert cheap_choice(state, legal) == "L0002"


@pytest.mark.parametrize("cards,level,expected", [
    (["S2", "SA", "SB", "HR"], "2", "L0001"),
    (["HR", "SB", "S2"], "2", "L0002"),
    (["HR", "SB"], "2", "L0001"),
    (["H2", "D2"], "2", "L0001"),
])
def test_card_cost_respects_level_jokers_and_preserves_heart_wildcard(cards, level, expected):
    state, legal = make_state_and_legal(
        cards, [("Single", card[1], (card,)) for card in cards], level=level, leader=None,
    )
    assert cheap_choice(state, legal) == expected


@pytest.mark.parametrize("stage,expected", [("tribute", "L0000"), ("back", "L0002")])
def test_tribute_and_back_have_their_own_order_without_play_heuristics(stage, expected):
    cards = ["HR", "SB", "SA", "S2"]
    state, legal = make_state_and_legal(
        cards, [(stage, stage, (card,)) for card in cards], stage=stage, level="2", leader=2,
    )
    result = asyncio.run(DecisionPolicy().decide(decision_input(state, legal), UnusableCalls()))
    assert cheap_choice(state, legal) == expected
    assert result.proposal.action_id == expected
    assert result.proposal.source == "rule"


def test_cheap_choice_checks_duplicate_card_ownership_before_context_exists():
    state, legal = make_state_and_legal(["S9", "D9"], [("Pair", "9", ("S9", "S9"))])
    with pytest.raises(ValueError, match="own hand"):
        cheap_choice(state, legal)


def test_cheap_choice_rejects_wrong_stage_action():
    state, legal = make_state_and_legal(["S9"], [("Single", "9", ("S9",))], stage="tribute")
    with pytest.raises(ValueError, match="stage"):
        cheap_choice(state, legal)


def test_wrong_binding_is_rejected_before_unique_or_fallback():
    state, legal = make_state_and_legal(["S9"], [("Single", "9", ("S9",))])
    valid_input = decision_input(state, legal)
    other_legal = legal.model_copy(update={"binding": legal.binding.model_copy(update={"state_version": 2})})
    with pytest.raises(ValueError, match="binding"):
        cheap_choice(state, other_legal)
    # Deliberately bypass DTO validation to verify the policy's final boundary.
    forged_context = valid_input.context.model_copy(update={"legal": other_legal})
    result = asyncio.run(DecisionPolicy().decide(valid_input.model_copy(update={"context": forged_context})))
    assert result.status == "invalidated"
    assert result.proposal is None
    assert result.error.code == "STATE_CONFLICT"


@pytest.mark.parametrize("bad_result", ["L9999", PASS, None, 42])
def test_invalid_chooser_result_uses_existing_fallback_without_inventing_pass(bad_result):
    state, legal = make_state_and_legal(
        ["S9", "D4"], [("Single", "9", ("S9",)), ("Single", "4", ("D4",))], leader=None,
    )
    result = asyncio.run(DecisionPolicy(lambda *_: bad_result).decide(
        decision_input(state, legal, fallback="L0001"), UnusableCalls(),
    ))
    assert result.proposal.action_id == "L0001"
    assert result.proposal.source == "fallback"
    assert result.proposal.cause_code == "TOOL_INVALID_RESULT"
    assert result.error.code == "TOOL_INVALID_RESULT"


def test_rule_exception_is_sanitized_and_distinguished_from_normal_rule_choice():
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS])

    def fail(*_):
        raise RuntimeError("sensitive provider text must not be copied")

    input = decision_input(state, legal, fallback="L0001")
    failed = asyncio.run(DecisionPolicy(fail).decide(input, UnusableCalls()))
    normal = asyncio.run(DecisionPolicy().decide(input, UnusableCalls()))
    assert failed.proposal.source == "fallback"
    assert failed.proposal.action_id == "L0001"
    assert failed.proposal.cause_code == "INTERNAL_ERROR"
    assert "sensitive" not in failed.model_dump_json()
    assert normal.proposal.source == "rule"
    assert normal.proposal.action_id == "L0000"
    assert normal.error is None
    assert normal.proposal.cause_code is None
    assert DecisionPolicy.rule_version == CHEAP_RULE_VERSION == "b02-cheap-v1"


def test_cancelled_rule_does_not_create_a_new_fallback_suggestion():
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS])

    def cancelled(*_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(DecisionPolicy(cancelled).decide(decision_input(state, legal)))


@pytest.mark.parametrize("policy", ["V1", "V2"])
def test_nonunique_future_policy_is_explicitly_unavailable(policy):
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS])
    result = asyncio.run(DecisionPolicy().decide(
        decision_input(state, legal, policy=policy), UnusableCalls(),
    ))
    assert result.status == "unavailable"
    assert result.proposal is None
    assert result.error.code == "POLICY_UNAVAILABLE"


def test_missing_validated_fallback_cannot_be_turned_into_a_proposal():
    state, legal = make_state_and_legal(["S9", "D4"], [("Single", "9", ("S9",)), PASS])
    valid = decision_input(state, legal)
    forged = valid.model_copy(update={"context": valid.context.model_copy(update={"fallback_action_id": "L9999"})})
    result = asyncio.run(DecisionPolicy().decide(forged))
    assert result.status == "unavailable"
    assert result.proposal is None
    assert result.error.code == "INVALID_REQUEST"
