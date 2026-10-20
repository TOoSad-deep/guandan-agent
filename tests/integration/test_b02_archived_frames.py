"""Parse historical frames independently; this does not run a game/state reducer."""

from collections import Counter
import json
from pathlib import Path

import pytest

from guandan_agent.contracts import LegalActionSet, RequestBinding, SeatConnection, VisibleState
from guandan_agent.hand import cheap_choice
from guandan_agent.platform import parse_platform_frame

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/phase3/results/probe_platform_run_02/normal"


@pytest.mark.parametrize("seat", range(4))
def test_historical_frames_parse_and_rules_keep_original_actions(seat):
    rows = [json.loads(line) for line in (ARCHIVE / f"seat-{seat}.jsonl").read_text().splitlines()]
    admission = next(row["message"] for row in rows if row["direction"] == "receive"
                     and row["message"].get("type") in ("CREATE_ROOM", "JOIN_ROOM"))
    session_id = f"00000000-0000-4000-8000-{seat + 30:012d}"
    connection = SeatConnection(url="ws://127.0.0.1:18181", mode="create" if seat == 0 else "join",
                                room_id=None if seat == 0 else admission["data"]["roomId"],
                                user_name=f"offline-{seat}", seat=seat, session_id=session_id,
                                connection_epoch=1, server_rounds=1)
    counts = Counter()
    for rx_seq, row in enumerate((row for row in rows if row["direction"] == "receive"), 1):
        raw = row["message"]
        event = parse_platform_frame(raw, config=connection, rx_seq=rx_seq,
                                     received_at="2026-10-19T00:00:00Z", received_mono=float(rx_seq),
                                     room_id=admission["data"]["roomId"])
        assert event.kind != "unknown", (seat, row["sequence"], raw.get("type"), raw.get("stage"))
        counts[event.kind] += 1
        if event.kind != "action_requested":
            continue
        data = event.model_dump(mode="json")["payload"]
        binding = RequestBinding(run_id="00000000-0000-4000-8000-000000000031",
                                 match_id="00000000-0000-4000-8000-000000000032",
                                 round_id=1, session_id=session_id, seat=seat, connection_epoch=1,
                                 request_id=f"{session_id}:q{counts['action_requested']}", state_version=rx_seq)
        legal = LegalActionSet(binding=binding, **data["parsed_actions"])
        ranks = [None, None]
        ranks[seat % 2], ranks[1 - seat % 2] = data["selfRank"], data["oppoRank"]
        remaining = [None] * 4
        remaining[seat] = len(data["handCards"])
        if "publicInfo" in data:
            remaining = [item["rest"] for item in data["publicInfo"]]
        leading = data.get("greaterAction") if data.get("greaterPos", -1) >= 0 else None
        # Request-local projection only; round_id/history are test scaffolding.
        state = VisibleState(binding=binding, stage=data["stage"], self_hand=data["handCards"],
                             level_rank=data["curRank"], remaining_counts=remaining, team_ranks=ranks,
                             turn_seat=seat, leader_seat=data["greaterPos"] if leading else None,
                             leading_action=leading, finished_order=[], history=[],
                             history_complete=False, history_omitted=0, facts_status="trusted")
        chosen_id = cheap_choice(state, legal)
        chosen = next(action for action in legal.actions if action.action_id == chosen_id)
        assert chosen.model_dump(mode="json")["wire_action"] == raw["actionList"][chosen.source_indices[0]]
    assert counts["match_result"] == counts["match_closed"] == 1
    print(json.dumps({"seat": seat, "historical_receive_frames": sum(counts.values()),
                      "counts": dict(counts), "new_games": 0, "model_calls": 0}, sort_keys=True))
