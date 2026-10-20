"""Regression for exact counts, isolated later summaries and honest display."""
import hashlib
import json

import pytest

from guandan_agent.config import RunConfig
from guandan_agent.contracts import PublicPlay
from guandan_agent.hand.summary import candidate_effect, hand_summary, remaining_cards
from guandan_agent.model.gateway import build_messages
from guandan_agent.offline.review import PROMPT_VERSION, _focused_input, _messages, _select, verify_program_facts
from guandan_agent.recording.query import _detail
from tests.unit.test_model_review import role_input
from tests.unit.test_offline_review import Calls, load_source, mutate, run


@pytest.mark.parametrize("level,count", [("3", 0), ("2", 1)])
def test_actual_wildcards_and_outgoing_rank_counts_do_not_predict_exchange(level, count):
    hand = ("H2", "D2", "S3", "D3", "HR", "HR")
    summary = hand_summary(hand, level)
    assert summary["wildcard_count"] == count
    assert summary["wildcards_in_hand"] == (["H2"] if count else [])
    assert summary["rank_counts"]["HR"] == 2
    effect = candidate_effect(hand, ("back", "back", ("D2",)), "back", level)
    assert effect["rank_count_changes"] == [{"rank": "2", "before": 2, "after_outgoing": 1}]
    assert effect["projected_hand_after_play"] is None
    assert effect["result_scope"] == "outgoing_only_exchange_result_unknown"
    assert "breaks_existing_pair" not in effect


def test_three_pairs_are_six_cards_and_no_sevens_or_eights_remain():
    hand = ("H6", "C6", "D6", "H7", "D7", "H8", "C8", "HR", "SB")
    action = ("ThreePair", "6", ("C6", "D6", "H7", "D7", "H8", "C8"))
    effect = candidate_effect(hand, action, "play", "2")
    assert effect["outgoing_count"] == 6
    assert effect["rank_count_changes"] == [
        {"rank": "6", "before": 3, "after_outgoing": 1},
        {"rank": "7", "before": 2, "after_outgoing": 0},
        {"rank": "8", "before": 2, "after_outgoing": 0},
    ]
    assert effect["projected_hand_after_play"]["card_counts"] == {"H6": 1, "HR": 1, "SB": 1}
    assert remaining_cards(("S3", "S3"), ("Single", "3", ("S3",))) == ("S3",)
    with pytest.raises(ValueError):
        remaining_cards(("S3",), ("Pair", "3", ("S3", "S3")))
    assert remaining_cards(hand, ("PASS", "PASS", "PASS")) == hand


@pytest.mark.parametrize("role", ["main", "review", "repair"])
def test_current_online_projection_bounds_history_and_supplies_exact_effects(role):
    history = tuple(PublicPlay(event_id=f"old-event-{i}", seat=1, action=("Single", "3", ("S3",))) for i in range(12))
    value = role_input(role, history=history)
    messages, size, omitted = build_messages(value, max_input_bytes=24576)
    payload = json.loads(messages[-1].content)
    assert omitted == 6 and size <= 24576
    assert payload["hand_summary"]["wildcards_in_hand"] == []
    assert len(payload["visible"]["history"]) == len(payload["semantic_facts"]["history"]) == 6
    assert "old-event-0" not in messages[-1].content
    for candidate in payload["candidates"]:
        effect = candidate["effect"]
        assert effect["outgoing_count"] + effect["projected_hand_after_play"]["card_count"] == 2


def test_current_offline_input_removes_old_reasons_and_later_card_faces():
    source = load_source()
    def polluted(rows):
        for row in rows:
            if row["event_type"] == "proposal_ready":
                row["payload"]["proposal"]["reason"] = "OLD_REASON_MUST_NOT_RETURN"
            event = row["payload"].get("event", {})
            if event.get("kind") == "play_observed":
                event["payload"].update(curPos=3, curAction=["Single", "3", ["H3"]])
    source = mutate(source, polluted)
    report, calls = run(source, prompt_version=PROMPT_VERSION)
    assert report.status == "completed" and verify_program_facts(source, report.items[0])
    projected = calls.inputs[0]
    assert "recorded_judgment" not in projected and "model_judgments" not in projected
    assert "OLD_REASON_MUST_NOT_RETURN" not in calls.messages[0][-1].content
    assert "effect" in projected["recorded_decision"]
    for row in projected["later_public_facts"]:
        assert set(row) == {"event_id", "kind", "summary"}
        assert "H3" not in json.dumps(row)
    assert "红桃3" not in "".join(projected["program_facts"]["later_facts"])
    assert any("座位3（对手）" in text for text in projected["program_facts"]["later_facts"])
    # Full source facts remain available in the final report, not rewritten.
    assert any("红桃3" in text for text in report.items[0].later_facts)


def test_current_compact_input_preserves_final_facts_and_rejects_omitted_citations():
    source = load_source()
    full, _ = run(source, prompt_version=PROMPT_VERSION)
    selected, _, _ = _select(source, None, None)
    messages = _messages([_focused_input(source, row, compact=True) for row in selected], prompt_version=PROMPT_VERSION)
    size = len(json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False, separators=(",", ":")).encode())
    config = RunConfig(model={"max_input_bytes": size})
    compact, calls = run(source, config=config, prompt_version=PROMPT_VERSION)
    assert compact.status == "partial" and len(calls.messages) == 1
    assert compact.items[0].later_facts == full.items[0].later_facts
    assert verify_program_facts(source, compact.items[0])
    omitted_id = full.items[0].evidence_event_ids[-1]
    def cite_missing(output, _):
        output["items"][0]["evidence_event_ids"].append(omitted_id)
    bad, _ = run(source, Calls(cite_missing), config=config, prompt_version=PROMPT_VERSION)
    assert bad.status == "failed"


def test_program_explanation_does_not_replace_or_validate_model_reason():
    source = load_source()
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
    before = _detail(source.cached_events, anchor.request_id, True)
    def change_reason(rows):
        for row in rows:
            if row["event_type"] == "proposal_ready":
                row["payload"]["proposal"]["reason"] = "错误模型原文必须保留"
    after = _detail(mutate(source, change_reason).cached_events, anchor.request_id, True)
    assert after.program_explanation == before.program_explanation and after.program_explanation
    assert after.proposal.reason == "错误模型原文必须保留"
    assert "错误模型原文" not in "".join(after.program_explanation)


@pytest.mark.parametrize("role,expected", [
    ("main", "be3ac0c1fb51c90ecb1482fb53672e0b5fca0a47065a3991c3c27fee747b0c25"),
    ("review", "884a0b41cb76b4aac1fb2c148b8b3eef794055f2fd86a31c859f60da6be8941e"),
    ("repair", "6c1ddcd349cf5abd905f952338fc9bba003e462c11c6f7b0678be096388a0b65"),
])
def test_v2_online_bytes_match_pre_change_snapshot(role, expected):
    messages = build_messages(role_input(role), max_input_bytes=24576, prompt_version="b09-facts-v2")[0]
    raw = json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == expected


def test_v2_offline_bytes_match_pre_change_snapshot():
    _, calls = run(load_source(), prompt_version="b09-offline-v2")
    raw = json.dumps([m.model_dump(mode="json") for m in calls.messages[0]], ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == "483638a7a1793ddaac9f26bfb55e025124641cf1ba86e675fad41ed598fe2dea"
