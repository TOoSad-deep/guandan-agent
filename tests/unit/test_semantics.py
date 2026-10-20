"""Program-owned semantics and versioned prompts; no provider traffic."""

import hashlib
import json

import pytest

from guandan_agent.contracts import PublicPlay, TributeContext
from guandan_agent.model.gateway import build_messages
from guandan_agent.semantics import describe_action, describe_card, describe_public, describe_state
from tests.unit.test_model_review import role_input


def test_card_decoding_distinguishes_jokers_and_only_current_heart_wildcard():
    assert describe_card("SB", "2") == "小王（SB）"
    assert describe_card("HR", "2") == "大王（HR）"
    assert describe_card("H2", "2") == "红桃2（H2，本局配牌）"
    assert describe_card("H2", "3") == "红桃2（H2）"
    assert describe_card("S2", "2") == "黑桃2（S2）"
    assert describe_card("HT", "2") == "红桃10（HT）"


def test_action_type_comes_from_wire_type_and_preserves_duplicate_faces():
    action = ("TwoTrips", "2", ("S2", "D2", "D2", "H3", "C3", "D3"))
    label = describe_action(action, "2")
    assert label.startswith("钢板（两个连续三张）")
    assert "共6张" in label and label.count("方块2（D2）") == 2
    assert "配牌" not in label
    assert describe_action(("ThreeWithTwo", "3", ("S3", "D3", "C3", "SA", "HA")), "2").startswith("三带二")
    assert describe_action(("back", "back", ("D2",)), "3").startswith("还贡")
    assert describe_action(("tribute", "tribute", ("CA",)), "5").startswith("进贡")


def test_state_keeps_unknown_counts_duplicate_hand_and_exact_relations():
    state = role_input("main").visible.model_copy(update={
        "stage": "back", "self_hand": ("C3", "C3", "HR", "SB"),
        "level_rank": "3", "remaining_counts": (4, None, None, None),
        "team_ranks": ("5", None),
    })
    facts = describe_state(state)
    assert facts["stage_label"] == "还贡" and "不是普通出牌" in facts["stage_note"]
    assert facts["seat_roles"] == {"self": 0, "partner": 2, "opponents": [1, 3], "self_team": 0}
    assert facts["self_hand_count"] == 4 and facts["self_hand_cards"].count("梅花3（C3）") == 2
    assert [row["count"] for row in facts["remaining_counts"]] == [4, None, None, None]
    assert facts["team_level_labels"] == ["队伍0级别为5（不是名次）", None]
    assert facts["wildcard_card_code"] == "H3"
    json.dumps(facts)


def test_notify_describes_current_action_and_pre_action_pass_sentinel():
    facts = {"curPos": 0, "curAction": ["Single", "B", ["SB"]],
             "greaterPos": -1, "greaterAction": ["PASS", "PASS", "PASS"]}
    sentences = describe_public("play_observed", facts, 0, "2")
    assert "座位0（本人）动作为" in sentences[0] and "小王" in sentences[0]
    assert "发生前无领先动作" in sentences[1] and "不表示其他三家连续不出" in sentences[1]
    facts.update(curPos=1, greaterPos=2, greaterAction=["Single", "8", ["C8"]])
    sentences = describe_public("play_observed", facts, 0, "2")
    assert "座位1（对手）" in sentences[0]
    assert "发生前领先者为座位2（队友）" in sentences[1]


def test_settlement_labels_finish_order_level_and_winner_separately():
    result = describe_public("round_result", {"order": [0, 1, 3, 2], "curRank": "2"}, 0, "2")
    assert "完成顺序（第1至第4）" in result[0]
    assert "级牌点数" in result[1] and "不是玩家名次" in result[1]
    facts = {"victory": 1, "victoryRank": ["8", "A"]}
    even = describe_public("match_result", facts, 0, "2")
    odd = describe_public("match_result", facts, 3, "2")
    assert "队伍1（座位1、3）获胜，本人所在队告负" in even[0]
    assert "本人所在队获胜" in odd[0]
    assert "队伍0为8、队伍1为A" in even[1] and "不是玩家名次" in even[1]


@pytest.mark.parametrize("role", ["main", "review", "repair"])
def test_new_prompt_supplies_semantics_for_all_roles_without_changing_candidates(role):
    input = role_input(role)
    old = json.loads(build_messages(input, max_input_bytes=24576, prompt_version="b07-review-v1")[0][1].content)
    messages, _, _ = build_messages(input, max_input_bytes=24576)
    new = json.loads(messages[1].content)
    assert new["prompt_version"] == "b09-facts-v3"
    assert new["semantic_facts"]["stage_label"] == "出牌"
    assert new["semantic_facts"]["self_hand_cards"] == ["黑桃9（S9）", "方块4（D4）"]
    for old_candidate, candidate in zip(old["candidates"], new["candidates"]):
        assert candidate.pop("description")
        assert candidate.pop("effect")
        assert candidate == old_candidate
    assert "SB 是小王，HR 是大王" in messages[0].content


def test_new_prompt_explains_back_context_without_claiming_exchange_completed():
    input = role_input("main")
    input = input.model_copy(update={
        "visible": input.visible.model_copy(update={"stage": "back", "level_rank": "3"}),
        "platform_context": TributeContext(tribute_pos=2, tribute_card="SB"),
    })
    payload = json.loads(build_messages(input, max_input_bytes=24576)[0][1].content)
    exchange = payload["semantic_facts"]["exchange_context"]
    assert exchange["tribute_from_seat"] == 2 and exchange["tribute_card"] == "小王（SB）"
    assert "返还给该进贡者" in exchange["instruction"]
    assert "交换后的手牌和余牌数不能" in payload["semantic_facts"]["stage_note"]


@pytest.mark.parametrize("role", ["main", "review", "repair"])
def test_history_trimming_removes_the_same_evidence_from_raw_and_semantics(role):
    history = tuple(PublicPlay(event_id=f"unique-evidence-{index}", seat=1,
                              action=("Single", "4", ("S4",))) for index in range(5))
    input = role_input(role, history=history)
    _, size, _ = build_messages(input, max_input_bytes=24576)
    messages, _, omitted = build_messages(input, max_input_bytes=size - 100)
    payload = json.loads(messages[1].content)
    assert omitted > 0
    raw, decoded = payload["visible"], payload["semantic_facts"]
    assert [row["event_id"] for row in raw["history"]] == [row["event_id"] for row in decoded["history"]]
    assert decoded["history_omitted"] == raw["history_omitted"] == omitted
    assert decoded["history_complete"] is raw["history_complete"] is False
    for event in history[:omitted]:
        assert event.event_id not in messages[1].content


@pytest.mark.parametrize("role,expected", [
    ("main", "26dac0d55d28b9b07619b50a79d4f0085a2c5bd1d8b40c869059fbeadfbc165c"),
    ("review", "65eadafbc18cbe1ff717fd7fad10f47b2a8f3cf10240ffdb521c5834350ebdbf"),
    ("repair", "ca9a07fc0891a4543c38d326e6b75f3625a96a46897931226fb40ad8164b91f3"),
])
def test_b07_prompt_payload_remains_byte_identical_for_each_role(role, expected):
    messages, _, _ = build_messages(role_input(role), max_input_bytes=24576, prompt_version="b07-review-v1")
    raw = json.dumps([message.model_dump(mode="json") for message in messages],
                     ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == expected
