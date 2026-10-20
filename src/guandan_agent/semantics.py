"""Deterministic descriptions of validated, public Guandan evidence.

These helpers translate wire codes; they do not predict hidden cards, infer
unobserved exchanges, or assess a strategy. Callers own evidence validation and
must pass only the history/public events retained in their bounded projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from guandan_agent.contracts import VisibleState, WireAction
from guandan_agent.hand.summary import candidate_effect, hand_summary


_SUITS = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块"}
_STAGES = {"play": "出牌", "tribute": "进贡", "back": "还贡"}
_TYPES = {
    "Single": "单张", "Pair": "对子", "Trips": "三张", "ThreeWithTwo": "三带二",
    "TwoTrips": "钢板（两个连续三张）", "ThreePair": "三个连续对子",
    "Straight": "顺子", "StraightFlush": "同花顺", "Bomb": "炸弹",
    "FourKings": "四王炸", "tribute": "进贡", "back": "还贡",
}


def _rank(rank: str) -> str:
    return "10" if rank == "T" else rank


def _relation(seat: int, self_seat: int) -> str:
    return "本人" if seat == self_seat else "队友" if seat == (self_seat + 2) % 4 else "对手"


def _seat(seat: int, self_seat: int) -> str:
    return f"座位{seat}（{_relation(seat, self_seat)}）"


def describe_card(code: str, level_rank: str) -> str:
    """Decode one validated card, retaining its code and wildcard meaning."""
    if code == "SB":
        return "小王（SB）"
    if code == "HR":
        return "大王（HR）"
    suit, rank = code[0], code[1:]
    if suit not in _SUITS or rank not in "23456789TJQKA" or len(rank) != 1:
        raise ValueError("Unknown card code.")
    special = "，本局配牌" if suit == "H" and rank == level_rank else ""
    return f"{_SUITS[suit]}{_rank(rank)}（{code}{special}）"


def describe_action(wire_action: WireAction, level_rank: str) -> str:
    """Use the authoritative action type; never reverse-engineer its cards."""
    kind, rank, cards = wire_action
    if kind == "PASS":
        return "不出（PASS）"
    name = _TYPES.get(kind, f"未解释牌型 {kind}")
    rank_text = "" if kind in {"tribute", "back", "FourKings"} else f"，平台牌型点数 {_rank(rank)}"
    faces = "、".join(describe_card(card, level_rank) for card in cards)
    return f"{name}{rank_text}，共{len(cards)}张：{faces}"


def describe_state(state: VisibleState) -> dict[str, object]:
    """Translate an already filtered VisibleState without adding hidden facts."""
    seat = state.binding.seat
    leader = None if state.leader_seat is None else _seat(state.leader_seat, seat)
    if state.stage == "play":
        stage_note = "选择出牌或不出；只能从本请求候选中选择。"
    else:
        stage_note = "这是贡还交换阶段，不是普通出牌；交换后的手牌和余牌数不能由本次交出一张直接推定。"
    return {
        "stage_label": _STAGES[state.stage],
        "stage_note": stage_note,
        "seat_roles": {"self": seat, "partner": (seat + 2) % 4,
                       "opponents": [other for other in range(4) if other % 2 != seat % 2],
                       "self_team": seat % 2},
        "team_members": [[0, 2], [1, 3]],
        "self_hand_count": len(state.self_hand),
        "self_hand_cards": [describe_card(card, state.level_rank) for card in state.self_hand],
        "wildcard_card_code": "H" + state.level_rank,
        "level_rank_label": f"本局级牌点数为{_rank(state.level_rank)}；级牌点数不是完成名次。",
        "team_level_labels": [None if rank is None else f"队伍{team}级别为{_rank(rank)}（不是名次）"
                              for team, rank in enumerate(state.team_ranks)],
        "remaining_counts": [{"seat": other, "relation": _relation(other, seat), "count": count}
                             for other, count in enumerate(state.remaining_counts)],
        "leader_label": leader,
        "leading_action_label": (None if state.leading_action is None
                                 else describe_action(state.leading_action, state.level_rank)),
        "finished_order_label": ("已记录的完成顺序：" + "、".join(_seat(other, seat) for other in state.finished_order)
                                 if state.finished_order else "当前未记录完成顺序。"),
        "history": [{"event_id": event.event_id,
                     "description": f"{_seat(event.seat, seat)}动作为：{describe_action(event.action, state.level_rank)}"}
                    for event in state.history],
        "history_complete": state.history_complete,
        "history_omitted": state.history_omitted,
        "facts_status": state.facts_status,
    }


def describe_public(kind: str, facts: Mapping[str, Any], self_seat: int,
                    level_rank: str) -> tuple[str, ...]:
    """Translate validated play/settlement projections, without other hands.

    Room.java sends curAction before replacing lastCard. Thus greaterAction is
    the leader *before* that action; its PASS sentinel is not three passes.
    """
    if kind == "play_observed":
        action = f"公开通知：{_seat(facts['curPos'], self_seat)}动作为：{describe_action(facts['curAction'], level_rank)}。"
        if facts["greaterPos"] == -1:
            prior = "该动作发生前无领先动作；greaterAction 的空值或 PASS 占位不表示其他三家连续不出。"
        else:
            prior = (f"该动作发生前领先者为{_seat(facts['greaterPos'], self_seat)}，"
                     f"领先动作为：{describe_action(facts['greaterAction'], level_rank)}。")
        return action, prior
    if kind == "round_result":
        order = "、".join(_seat(seat, self_seat) for seat in facts["order"])
        return (f"本局结算完成顺序（第1至第4）：{order}。",
                f"本局级牌点数为{_rank(facts['curRank'])}；curRank 是级牌点数，不是玩家名次。")
    if kind == "match_result":
        team = facts["victory"]
        result = "获胜" if team == self_seat % 2 else "告负"
        return (f"整场结算：队伍{team}（座位{team}、{team + 2}）获胜，本人所在队{result}。",
                f"结算双方级别：队伍0为{_rank(facts['victoryRank'][0])}、队伍1为{_rank(facts['victoryRank'][1])}；victoryRank 是队伍级别，不是玩家名次。")
    raise ValueError("Unsupported public event kind.")


def public_summary(kind: str, facts: Mapping[str, Any], self_seat: int, level_rank: str) -> tuple[str, ...]:
    """Keep public outcomes but omit later players' individual card faces."""
    if kind != "play_observed":
        return describe_public(kind, facts, self_seat, level_rank)

    def action_summary(action):
        if action[0] == "PASS":
            return "不出（PASS）"
        return f"{_TYPES.get(action[0], action[0])}，平台点数{_rank(action[1])}，共{len(action[2])}张"

    current = f"事后公开动作：{_seat(facts['curPos'], self_seat)}，{action_summary(facts['curAction'])}。"
    prior = ("该动作之前无领先动作；PASS占位不证明随后无人压牌。" if facts["greaterPos"] == -1
             else f"该动作之前领先者：{_seat(facts['greaterPos'], self_seat)}，{action_summary(facts['greaterAction'])}。")
    return current, prior


def describe_candidate_effect(state: VisibleState, action: WireAction) -> tuple[str, ...]:
    """Display source-derived arithmetic separately from any model reason."""
    effect = candidate_effect(state.self_hand, action, state.stage, state.level_rank)
    own = hand_summary(state.self_hand, state.level_rank)
    lines = [f"记录动作：{describe_action(action, state.level_rank)}。",
             f"请求时本人配牌为{own['wildcard_card']}，实际持有{own['wildcard_count']}张。"]
    if effect["rank_count_changes"]:
        changes = "；".join(f"点数{row['rank']}：{row['before']}→{row['after_outgoing']}张"
                           for row in effect["rank_count_changes"])
        lines.append(f"仅交出这些牌的计数变化：{changes}。这不评价组合优劣。")
    if state.stage in {"tribute", "back"}:
        lines.append("贡还涉及换牌；最终手牌和张数不能由本次交出牌推定。")
    else:
        lines.append(f"仅按该候选扣减，请求时手牌将剩{effect['projected_hand_after_play']['card_count']}张；不代表已确认执行。")
    return tuple(lines)
