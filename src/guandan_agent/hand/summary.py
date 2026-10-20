"""Exact multiset arithmetic shared by candidate scoring and prompt summaries.

Counts describe physical cards, not optimal combinations or wildcard assignments.
"""
from collections import Counter

from guandan_agent.contracts import WireAction


def rank_counts(cards: tuple[str, ...]) -> Counter[str]:
    return Counter(card if card in ("SB", "HR") else card[1] for card in cards)


def remaining_cards(hand: tuple[str, ...], action: WireAction) -> tuple[str, ...]:
    outgoing = Counter() if action[0] == "PASS" else Counter(action[2])
    if outgoing - Counter(hand):
        raise ValueError("Action consumes cards outside the own hand")
    result = []
    for card in hand:
        if outgoing[card]:
            outgoing[card] -= 1
        else:
            result.append(card)
    return tuple(result)


def hand_summary(hand: tuple[str, ...], level_rank: str) -> dict:
    wildcards = [card for card in hand if card == "H" + level_rank]
    return {
        "card_count": len(hand),
        "card_counts": dict(sorted(Counter(hand).items())),
        "rank_counts": dict(sorted(rank_counts(hand).items())),
        "wildcard_card": "H" + level_rank,
        "wildcards_in_hand": wildcards,
        "wildcard_count": len(wildcards),
    }


def candidate_effect(hand: tuple[str, ...], action: WireAction, stage: str, level_rank: str) -> dict:
    """Known outgoing effects; a future exchange's final hand stays unknown."""
    remaining = remaining_cards(hand, action)
    before, after = rank_counts(hand), rank_counts(remaining)
    outgoing = () if action[0] == "PASS" else action[2]
    exchange = stage in {"tribute", "back"}
    return {
        "basis": "own_request_hand",
        "outgoing_card_counts": dict(sorted(Counter(outgoing).items())),
        "outgoing_count": len(outgoing),
        "rank_count_changes": [
            {"rank": rank, "before": count, "after_outgoing": after[rank]}
            for rank, count in sorted(before.items()) if after[rank] != count
        ],
        "result_scope": "outgoing_only_exchange_result_unknown" if exchange else "hypothetical_play",
        "projected_hand_after_play": None if exchange else hand_summary(remaining, level_rank),
    }
