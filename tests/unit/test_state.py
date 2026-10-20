"""Fact reduction and historical replay; these tests never start a new game."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from guandan_agent.contracts import RequestBinding, SeatConnection
from guandan_agent.platform import parse_platform_frame
from guandan_agent.runtime.state import StateConflict, TableState

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/phase3/results/probe_platform_run_02/normal"
SESSION = "00000000-0000-4000-8000-000000000041"


def connection(seat=0):
    return SeatConnection(url="ws://127.0.0.1:18181", mode="create" if seat == 0 else "join",
                          room_id=None if seat == 0 else 1, user_name=f"test-{seat}", seat=seat,
                          session_id=SESSION, connection_epoch=1, server_rounds=1)


def event(raw, seat=0, seq=1):
    result = parse_platform_frame(raw, config=connection(seat), rx_seq=seq,
                                  received_at="2026-10-19T00:00:00Z", received_mono=float(seq), room_id=1)
    assert result.kind != "unknown", result.payload
    return result


def apply(state, raw, seq=1):
    parsed = event(raw, state.seat, seq)
    state.apply(parsed, f"event-{seq}")
    return parsed


def begin(hand=("S2", "S2", "H4"), seat=0, level="2"):
    return {"type": "notify", "stage": "beginning", "myPos": seat,
            "handCards": list(hand), "levelCard": level}


def play(actor=0, cards=("S2",), action=None):
    return {"type": "notify", "stage": "play", "curPos": actor,
            "curAction": action or ["Single", "2", list(cards)],
            "greaterPos": -1, "greaterAction": ["PASS", "PASS", "PASS"]}


def request(hand=("S2", "S2", "H4"), *, stage="play", seat=0, counts=None, context=None,
            leader=None, level="2", self_rank="2", oppo_rank="2", tribute=(1, "SA")):
    raw = {"type": "act", "stage": stage, "handCards": list(hand), "curRank": level,
           "selfRank": self_rank, "oppoRank": oppo_rank,
           "actionList": [["Single", hand[0][1], [hand[0]]]] if stage == "play"
           else [[stage, stage, [hand[0]]]]}
    if stage == "play":
        counts = counts or [len(hand) if n == seat else 27 for n in range(4)]
        raw.update(publicInfo=[{"rest": count} for count in counts],
                   curPos=-1 if context is None else context[0],
                   curAction=None if context is None else context[1],
                   greaterPos=-1 if leader is None else leader[0],
                   greaterAction=None if leader is None else leader[1])
    if stage == "back":
        raw.update(tributePos=tribute[0], tribute=tribute[1])
    return raw


def binding(state, *, seat=None, round_id=None):
    return RequestBinding(run_id="00000000-0000-4000-8000-000000000042",
                          match_id="00000000-0000-4000-8000-000000000043",
                          round_id=round_id or state.round_id, session_id=SESSION,
                          seat=state.seat if seat is None else seat, connection_epoch=1,
                          request_id=f"{SESSION}:q1", state_version=1)


def settled(order=(0, 1, 2, 3), level="2"):
    return {"type": "notify", "stage": "episodeOver", "curRank": level,
            "order": list(order), "restCards": [[3, ["HR", "SA"]]]}


def signature(state):
    return (state.table_view(), state.history, state.history_complete, state.history_omitted,
            state.tribute_transfers, state.back_transfers, state.exchange_applied)


def test_beginning_clears_every_round_fact_but_initial_table_is_absent():
    state = TableState(0)
    assert state.table_view() is None
    apply(state, begin())
    assert state.remaining_counts == (3, None, None, None)
    apply(state, request(self_rank="4", oppo_rank="5"))
    apply(state, play())
    apply(state, settled())
    apply(state, begin(("CA", "SB"), level="3"))
    assert state.round_id == 2
    assert state.stage == "waiting"
    assert state.self_hand == ("CA", "SB")
    assert state.remaining_counts == (2, None, None, None)
    assert state.team_ranks == (None, None)
    assert state.level_rank == "3"
    assert state.finished_order == state.history == ()
    assert state.leader_seat is state.leading_action is None
    assert state.history_complete and state.history_omitted == 0
    assert state.tribute_transfers == state.back_transfers == ()
    assert not state.exchange_applied


def test_play_uses_multiset_and_act_context_does_not_repeat_action():
    state = TableState(0)
    apply(state, begin())
    apply(state, request())
    frame = play()
    apply(state, frame, 2)
    assert state.self_hand == ("S2", "H4")
    assert state.remaining_counts[0] == 2
    assert state.leader_seat == 0  # notify greaterPos=-1 was the PRE-update state
    apply(state, request(("H4", "S2"), counts=[2, 27, 27, 27],
                         context=(0, frame["curAction"]), leader=(0, frame["curAction"])), 3)
    assert state.self_hand == ("H4", "S2")
    assert len(state.history) == 1
    # Same value in another actual network observation remains a distinct play.
    apply(state, frame, 4)
    assert state.self_hand == ("H4",)
    assert len(state.history) == 2
    assert [entry.event_id for entry in state.history] == ["event-2", "event-4"]


def test_unknown_opponent_count_stays_unknown_until_authoritative_request():
    state = TableState(0)
    apply(state, begin())
    frame = play(3, ("SA", "SA"), ["Pair", "A", ["SA", "SA"]])
    apply(state, frame)
    assert state.remaining_counts == (3, None, None, None)
    apply(state, request(counts=[3, 27, 27, 25], context=(3, frame["curAction"])))
    assert state.remaining_counts == (3, 27, 27, 25)
    assert len(state.history) == 1


@pytest.mark.parametrize("late_exchange", [
    {"type": "notify", "stage": "tribute", "result": [[1, 0, "SA"]]},
    {"type": "notify", "stage": "anti-tribute", "antiNums": 1, "antiPos": [1]},
    request(("S2", "H4"), stage="tribute"),
])
def test_zero_history_window_does_not_erase_round_progress(late_exchange):
    state = TableState(0, history_limit=0)
    apply(state, begin())
    apply(state, play())
    assert state.history == () and state.history_omitted == 1
    before = signature(state)
    with pytest.raises(StateConflict):
        apply(state, late_exchange)
    assert signature(state) == before


def test_nonself_team_rank_order_and_current_level_are_distinct():
    state = TableState(1)
    apply(state, begin(seat=1, level="6"))
    apply(state, request(seat=1, level="6", self_rank="7", oppo_rank="5"))
    visible = state.visible(binding(state))
    assert visible.team_ranks == ("5", "7")
    assert visible.level_rank == "6"
    assert visible.turn_seat == 1


@pytest.mark.parametrize("hand,transfers,backs,expected", [
    (("SA", "SA", "H4"), [[0, 1, "SA"]], [[1, 0, "C2"]], ("SA", "H4", "C2")),
    (("C2", "S2", "H4"), [[1, 0, "SA"]], [[0, 1, "C2"]], ("S2", "H4", "SA")),
    (("S2", "S2", "H4"), [[3, 1, "SA"]], [[1, 3, "C2"]], ("S2", "S2", "H4")),
    (("SA", "S2", "H4"), [[0, 1, "SA"], [2, 3, "SB"]],
     [[1, 0, "C2"], [3, 2, "D2"]], ("S2", "H4", "C2")),
])
def test_tribute_waits_for_complete_back_mapping_then_exchanges_once(hand, transfers, backs, expected):
    state = TableState(0)
    apply(state, begin(hand))
    apply(state, {"type": "notify", "stage": "tribute", "result": transfers})
    assert state.self_hand == hand  # the announcement alone transfers nothing
    if transfers[0][1] == 0:
        apply(state, request(hand, stage="back", tribute=(transfers[0][0], transfers[0][2])))
        assert state.self_hand == hand  # back hand is also pre-exchange
    frame = {"type": "notify", "stage": "back", "result": backs, "count": len(hand)}
    apply(state, frame)
    assert Counter(state.self_hand) == Counter(expected)
    assert state.exchange_applied
    before = signature(state)
    with pytest.raises(StateConflict, match="pending tribute"):
        apply(state, frame, 2)
    assert signature(state) == before
    apply(state, request(tuple(reversed(expected))))
    assert Counter(state.self_hand) == Counter(expected)


@pytest.mark.parametrize("bad_back", [
    {"result": [[1, 2, "C2"]], "count": 3},
    {"result": [[1, 0, "C2"]], "count": 2},
])
def test_exchange_conflict_is_atomic(bad_back):
    state = TableState(0)
    apply(state, begin(("SA", "S2", "H4")))
    apply(state, {"type": "notify", "stage": "tribute", "result": [[0, 1, "SA"]]})
    before = signature(state)
    with pytest.raises(StateConflict):
        apply(state, {"type": "notify", "stage": "back", **bad_back})
    assert signature(state) == before


def test_back_cannot_spend_a_card_that_will_only_arrive_in_the_exchange():
    state = TableState(0)
    apply(state, begin(("S2", "H4")))
    apply(state, {"type": "notify", "stage": "tribute", "result": [[1, 0, "SA"]]})
    before = signature(state)
    with pytest.raises(StateConflict, match="absent own cards"):
        apply(state, {"type": "notify", "stage": "back", "result": [[0, 1, "SA"]], "count": 2})
    assert signature(state) == before


def test_anti_tribute_skips_exchange_and_rejects_contradictory_late_transfer():
    state = TableState(0)
    apply(state, begin())
    apply(state, {"type": "notify", "stage": "anti-tribute", "antiNums": 1, "antiPos": [1]})
    apply(state, request())
    assert not state.exchange_applied
    with pytest.raises(StateConflict):
        apply(state, {"type": "notify", "stage": "tribute", "result": [[1, 0, "SA"]]})


def test_absent_play_cards_and_negative_known_counts_fail_atomically():
    state = TableState(0)
    apply(state, begin())
    apply(state, request(counts=[3, 1, 27, 27]))
    for raw in (play(cards=("S2", "S2", "S2")), play(1, ("SA", "SA"))):
        before = signature(state)
        with pytest.raises(StateConflict):
            apply(state, raw)
        assert signature(state) == before


@pytest.mark.parametrize("bad_request", [
    request(("S2", "H4")),
    request(counts=[3, 26, 27, 27]),
    request(context=(1, ["Single", "A", ["SA"]])),
    request(level="3"),
])
def test_action_cannot_silently_repair_missing_or_conflicting_evidence(bad_request):
    state = TableState(0)
    apply(state, begin())
    apply(state, request())
    before = signature(state)
    with pytest.raises(StateConflict):
        apply(state, bad_request)
    assert signature(state) == before


def test_tribute_back_and_play_require_their_missing_prior_evidence():
    state = TableState(0)
    with pytest.raises(StateConflict, match="beginning"):
        apply(state, request())
    apply(state, begin())
    with pytest.raises(StateConflict, match="pairing"):
        apply(state, request(stage="back"))
    apply(state, {"type": "notify", "stage": "tribute", "result": [[1, 0, "SA"]]})
    with pytest.raises(StateConflict, match="completion"):
        apply(state, request())
    with pytest.raises(StateConflict, match="completion"):
        apply(state, play())


@pytest.mark.parametrize("limit", [0, 1])
def test_history_clipping_is_explicit_without_changing_current_facts(limit):
    state = TableState(0, history_limit=limit)
    apply(state, begin())
    apply(state, play(1), 1)
    apply(state, play(2, action=["PASS", "PASS", "PASS"]), 2)
    assert len(state.history) == limit
    assert state.history_omitted == 2 - limit
    assert not state.history_complete
    assert state.facts_status == "trusted"
    # Last public play is retained separately even when display history is clipped.
    apply(state, request(context=(2, ["PASS", "PASS", "PASS"])))
    visible = state.visible(binding(state))
    assert visible.history_omitted == 2 - limit


def test_settlement_reveals_do_not_enter_visible_state_or_mutate_prior_snapshot():
    state = TableState(0)
    apply(state, begin())
    apply(state, request())
    previous = state.visible(binding(state))
    previous_json = previous.model_dump_json()
    apply(state, settled())
    assert previous.model_dump_json() == previous_json
    assert "restCards" not in state.table_view().model_dump_json()
    assert "HR" not in previous_json and "SA" not in previous_json
    apply(state, begin())
    apply(state, request())
    current = state.visible(binding(state)).model_dump_json()
    assert "HR" not in current and "SA" not in current


def test_degradation_is_explicit_and_request_cannot_restore_it():
    state = TableState(0)
    apply(state, begin())
    state.mark_degraded()
    assert state.facts_status == "degraded" and not state.history_complete
    with pytest.raises(StateConflict, match="Degraded"):
        apply(state, request())


def test_visible_binding_rejects_other_seat_round_or_nonaction_stage():
    state = TableState(0)
    apply(state, begin())
    with pytest.raises(StateConflict, match="action stage"):
        state.visible(binding(state))
    apply(state, request())
    for wrong in (binding(state, seat=1), binding(state, round_id=2)):
        with pytest.raises(StateConflict, match="different seat or round"):
            state.visible(wrong)


@pytest.mark.parametrize("seat", range(4))
def test_complete_historical_match_reduces_all_four_seat_streams(seat):
    state = TableState(seat)
    counts = Counter()
    for line_no, line in enumerate((ARCHIVE / f"seat-{seat}.jsonl").read_text().splitlines(), 1):
        row = json.loads(line)
        if row["direction"] != "receive":
            continue
        parsed = event(row["message"], seat, counts.total() + 1)
        state.apply(parsed, f"historical-seat-{seat}-line-{line_no}")
        counts[parsed.kind] += 1
        if parsed.kind == "action_requested":
            visible = state.visible(binding(state))
            assert Counter(visible.self_hand) == Counter(row["message"]["handCards"])
            assert all(entry.event_id.startswith(f"historical-seat-{seat}-") for entry in visible.history)
    assert state.round_id == counts["round_result"] == 15
    assert counts["match_result"] == counts["match_closed"] == 1
    assert state.stage == "settled"


def test_t11_real_prefix_establishes_old_state_before_cross_round_reset():
    fixture = json.loads((ROOT / "tests/fixtures/runtime/state_t11.json").read_text())
    source = ROOT / fixture["source"]
    lines = source.read_text().splitlines()
    for anchor in fixture["anchors"]:
        assert hashlib.sha256(lines[anchor["line"] - 1].encode()).hexdigest() == anchor["sha256"]
    state = TableState(0)
    seen = 0
    for line_no, line in enumerate(lines[:144], 1):
        row = json.loads(line)
        if row["direction"] != "receive":
            continue
        seen += 1
        parsed = event(row["message"], seq=seen)
        state.apply(parsed, f"historical-line-{line_no}")
        if line_no == 136:
            assert state.remaining_counts == (1, 0, 0, 1)
        elif line_no == 138:
            assert state.remaining_counts == (0, 0, 0, 1)
        elif line_no == 140:
            assert state.round_id == 2
            assert state.remaining_counts == (27, None, None, None)
            assert state.team_ranks == (None, None)
            assert state.finished_order == state.history == ()
            assert state.leader_seat is state.leading_action is None
            assert state.history_complete and state.history_omitted == 0
        elif line_no == 143:
            assert state.remaining_counts == (27, None, None, None)
            assert len(state.history) == 1
        elif line_no == 144:
            assert state.remaining_counts == (27, 27, 27, 21)
            assert state.team_ranks == ("2", "3")
            assert state.level_rank == "3"
            assert len(state.history) == 1
            assert state.history[0].event_id == "historical-line-143"


def test_finished_player_cannot_pass_or_receive_another_request():
    state = TableState(0)
    apply(state, begin(("S2",)))
    apply(state, request(("S2",)))
    apply(state, play())
    assert state.finished_order == (0,)
    before = signature(state)
    with pytest.raises(StateConflict, match="Finished player"):
        apply(state, play(action=["PASS", "PASS", "PASS"]))
    raw = request(("S2",))
    raw["handCards"] = []
    raw["publicInfo"][0]["rest"] = 0
    raw["actionList"] = [["PASS", "PASS", "PASS"]]
    with pytest.raises(StateConflict, match="Finished player"):
        apply(state, raw)
    assert signature(state) == before


def test_pass_preserves_hand_and_public_lead_until_request_resets_it():
    state = TableState(0)
    apply(state, begin())
    apply(state, request())
    apply(state, play(1, ("SA",), ["Single", "A", ["SA"]]))
    before_hand, before_counts = state.self_hand, state.remaining_counts
    apply(state, play(2, action=["PASS", "PASS", "PASS"]))
    assert state.self_hand == before_hand and state.remaining_counts == before_counts
    assert state.leader_seat == 1
    assert state.leading_action == ("Single", "A", ("SA",))
    # A new lead is normalized only from the action request's current fields.
    apply(state, request(counts=[3, 26, 27, 27]))
    assert state.leader_seat is state.leading_action is None
