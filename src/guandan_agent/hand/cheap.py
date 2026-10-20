"""Small, deterministic B02 rules over the complete platform action list.

This is a temporary baseline, not a rules engine or a hand-search algorithm.
The platform supplies legality; this module checks ownership and picks an ID.
It needs neither RequestContext (whose fallback it creates) nor a model/Java
client. B05 replaces the normal V0 ranking with the final shared rule version.
"""

from collections import Counter

from guandan_agent.contracts import LegalAction, LegalActionSet, VisibleState


CHEAP_RULE_VERSION = "b02-cheap-v1"
_RANK_STRENGTH = {rank: index for index, rank in enumerate("23456789TJQKA", 2)}
_BOMB_TYPES = frozenset({"Bomb", "StraightFlush", "FourKings"})


def _check_input(state: VisibleState, legal: LegalActionSet) -> None:
    if state.binding != legal.binding:
        raise ValueError("state and legal actions must have identical binding")
    hand = Counter(state.self_hand)
    for action in legal.actions:
        kind, rank, cards = action.wire_action
        if state.stage in ("tribute", "back"):
            if kind != state.stage or rank != state.stage or len(cards) != 1:
                raise ValueError("action does not match the tribute/back stage")
        elif kind in ("tribute", "back"):
            raise ValueError("tribute/back action cannot appear in play")
        if kind != "PASS" and Counter(cards) - hand:
            raise ValueError("action consumes cards outside the own hand")


def _strength(card: str, level: str) -> int:
    # Level cards outrank A, below the small and large jokers. This is a
    # conservation cost, not an action-comparison implementation: sequences
    # and wildcard substitutions stay entirely under platform legality.
    if card == "HR":
        return 17
    if card == "SB":
        return 16
    if card[1] == level:
        return 15
    return _RANK_STRENGTH[card[1]]


def _wildcards(action: LegalAction, level: str) -> int:
    return action.wire_action[2].count(f"H{level}")


def _play_cost(action: LegalAction, level: str) -> tuple[int, int, int, int]:
    cards = action.wire_action[2]
    return (
        -len(cards),
        _wildcards(action, level),
        sum(_strength(card, level) for card in cards),
        action.source_indices[0],
    )


def _bomb_cost(action: LegalAction, level: str) -> tuple[bool, int, int, int, int]:
    cards = action.wire_action[2]
    return (
        action.wire_action[0] == "FourKings",
        len(cards),
        _wildcards(action, level),
        sum(_strength(card, level) for card in cards),
        action.source_indices[0],
    )


def cheap_choice(state: VisibleState, legal: LegalActionSet) -> str:
    """Return an existing ID without constructing a new action or context.

    Priorities are: complete-set singleton; stage-specific tribute/back;
    finish the hand; pass behind a partner; use a normal play; preserve bombs
    if a legal PASS exists and no opponent is known to have 1–2 cards.
    Ordinary play prefers more cards, then fewer wildcards/lower card cost.
    All ties retain the platform's first original index. Unknown counts stay
    unknown and cannot by themselves trigger the small-hand exception.
    """
    _check_input(state, legal)
    actions = legal.actions
    if len(actions) == 1:
        return actions[0].action_id

    if state.stage in ("tribute", "back"):
        direction = -1 if state.stage == "tribute" else 1
        selected = min(
            actions,
            key=lambda action: (
                direction * _strength(action.wire_action[2][0], state.level_rank),
                action.source_indices[0],
            ),
        )
        return selected.action_id

    non_pass = [action for action in actions if action.wire_action[0] != "PASS"]
    passing = next((action for action in actions if action.wire_action[0] == "PASS"), None)
    hand = Counter(state.self_hand)
    finishing = [action for action in non_pass if Counter(action.wire_action[2]) == hand]
    if finishing:
        return finishing[0].action_id

    partner = (state.binding.seat + 2) % 4
    if passing is not None and state.leader_seat == partner:
        return passing.action_id

    ordinary = [action for action in non_pass if action.wire_action[0] not in _BOMB_TYPES]
    if ordinary:
        return min(ordinary, key=lambda action: _play_cost(action, state.level_rank)).action_id

    opponent_close = any(
        count in (1, 2)
        for seat, count in enumerate(state.remaining_counts)
        if seat % 2 != state.binding.seat % 2
    )
    if passing is not None and state.leader_seat is not None and not opponent_close:
        return passing.action_id
    if non_pass:
        return min(non_pass, key=lambda action: _bomb_cost(action, state.level_rank)).action_id

    # The complete legal set is nonempty. Never synthesize a PASS: this branch
    # returns only the exact platform-provided option (normally singleton).
    return actions[0].action_id
