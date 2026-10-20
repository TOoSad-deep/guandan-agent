"""Deterministic B07 review triggers derived only from the current visible state."""

from guandan_agent.config import ReviewConfig
from guandan_agent.contracts import DecisionInput, ModelDecision


RISK_VERSION = "b07-risk-v1"
# These are the canonical types in the pinned OpenGuanDan action mapping.
BOMB_TYPES = frozenset({"Bomb", "StraightFlush", "FourKings"})


def detect_risks(input: DecisionInput, decision: ModelDecision,
                 config: ReviewConfig) -> tuple[str, ...]:
    """Return stable, merged triggers; no guessing unknown or finished hands.

    SPLIT_COST remains disabled until its scoring and threshold are validated.
    The frozen review configuration is an explicit dependency for that future
    extension; neither this function nor the graph reads mutable run settings.
    """
    state = input.context.state
    if state.stage != "play":
        return ()
    action = next((action for action in input.context.legal.actions
                   if action.action_id == decision.action_id), None)
    snapshot = input.candidates
    if (action is None or snapshot is None
            or decision.action_id not in {row.action_id for row in snapshot.candidates}):
        return ()
    triggers = []
    if any(seat % 2 != state.binding.seat % 2 and seat not in state.finished_order
           and count in (1, 2) for seat, count in enumerate(state.remaining_counts)):
        triggers.append("OPPONENT_NEAR_FINISH")
    if action.wire_action[0] in BOMB_TYPES:
        triggers.append("USE_BOMB")
    if (state.leader_seat == (state.binding.seat + 2) % 4
            and action.wire_action[0] != "PASS"):
        triggers.append("BEAT_PARTNER")
    return tuple(triggers)
