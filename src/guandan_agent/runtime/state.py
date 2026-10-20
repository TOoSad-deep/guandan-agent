"""Seat-local public-fact reduction, privately owned by exactly one Session.

This module does not own connection sequences, requests, tasks or submission
rights. Every event builds a new immutable fact snapshot before publishing it;
a rejected observation therefore cannot leave a partially deducted hand.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from guandan_agent.contracts import (
    PlatformEvent, PublicPlay, RequestBinding, TableView, VisibleState, WireAction,
)

_PASS = ("PASS", "PASS", "PASS")
Transfer = tuple[int, int, str]
TableStage = Literal["waiting", "play", "tribute", "back", "settled"]


class StateConflict(ValueError):
    """Safe invariant failure; contains no raw frame or hidden card details."""

    code = "STATE_CONFLICT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateConflict(message)


def _action(value: Any) -> WireAction:
    # The platform parser has already validated these immutable JSON arrays.
    return cast(WireAction, tuple(value))


def _same_action(left: WireAction, right: WireAction) -> bool:
    if left == _PASS or right == _PASS:
        return left == right
    return left[:2] == right[:2] and Counter(left[2]) == Counter(right[2])


def _remove(hand: tuple[str, ...], cards: tuple[str, ...]) -> tuple[str, ...]:
    wanted = Counter(cards)
    _require(not (wanted - Counter(hand)), "Observed action consumes absent own cards")
    kept: list[str] = []
    for card in hand:
        if wanted[card]:
            wanted[card] -= 1
        else:
            kept.append(card)
    return tuple(kept)


@dataclass(frozen=True)
class _Facts:
    round_id: int = 0
    stage: TableStage = "waiting"
    self_hand: tuple[str, ...] = ()
    remaining_counts: tuple[int | None, ...] = (None, None, None, None)
    team_ranks: tuple[str | None, ...] = (None, None)
    level_rank: str | None = None
    leader_seat: int | None = None
    leading_action: WireAction | None = None
    finished_order: tuple[int, ...] = ()
    history: tuple[PublicPlay, ...] = ()
    history_complete: bool = True
    history_omitted: int = 0
    facts_status: Literal["trusted", "degraded"] = "trusted"
    last_play: PublicPlay | None = None
    tribute_transfers: tuple[Transfer, ...] = ()
    back_transfers: tuple[Transfer, ...] = ()
    exchange_applied: bool = False
    anti_tribute: bool = False


class TableState:
    """Atomic fact helper; callers serialize all access through their Session."""

    def __init__(self, seat: int, history_limit: int = 100) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError("seat must be 0..3")
        if type(history_limit) is not int or history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer")
        self.seat = seat
        self.history_limit = history_limit
        self._facts = _Facts()

    @property
    def round_id(self) -> int:
        return self._facts.round_id

    @property
    def stage(self) -> TableStage:
        return self._facts.stage

    @property
    def self_hand(self) -> tuple[str, ...]:
        return self._facts.self_hand

    @property
    def remaining_counts(self) -> tuple[int | None, ...]:
        return self._facts.remaining_counts

    @property
    def team_ranks(self) -> tuple[str | None, ...]:
        return self._facts.team_ranks

    @property
    def level_rank(self) -> str | None:
        return self._facts.level_rank

    @property
    def leader_seat(self) -> int | None:
        return self._facts.leader_seat

    @property
    def leading_action(self) -> WireAction | None:
        return self._facts.leading_action

    @property
    def finished_order(self) -> tuple[int, ...]:
        return self._facts.finished_order

    @property
    def history(self) -> tuple[PublicPlay, ...]:
        return self._facts.history

    @property
    def history_complete(self) -> bool:
        return self._facts.history_complete

    @property
    def history_omitted(self) -> int:
        return self._facts.history_omitted

    @property
    def facts_status(self) -> Literal["trusted", "degraded"]:
        return self._facts.facts_status

    @property
    def tribute_transfers(self) -> tuple[Transfer, ...]:
        return self._facts.tribute_transfers

    @property
    def back_transfers(self) -> tuple[Transfer, ...]:
        return self._facts.back_transfers

    @property
    def exchange_applied(self) -> bool:
        return self._facts.exchange_applied

    def mark_degraded(self) -> None:
        """Preserve facts but make their loss of continuity explicit."""
        self._facts = replace(self._facts, facts_status="degraded", history_complete=False)

    def apply(self, event: PlatformEvent, event_id: str) -> None:
        """Apply one already parsed event, or fail without mutating any facts."""
        _require(bool(event_id), "A recorded public event requires an event ID")
        kind, data, old = event.kind, event.payload, self._facts
        if kind in ("joined", "join_failed", "auxiliary_notice", "match_result", "match_closed"):
            return
        if kind == "unknown":
            raise StateConflict("Unrecognized platform event cannot establish trusted facts")
        if kind == "round_started":
            _require(old.round_id == 0 or old.stage == "settled", "New round arrived before settlement")
            _require(data["myPos"] == self.seat, "Beginning belongs to a different seat")
            hand = tuple(data["handCards"])
            _require(0 < len(hand) <= 27, "Invalid beginning hand count")
            counts: list[int | None] = [None] * 4
            counts[self.seat] = len(hand)
            # A trusted beginning resets round-local facts only. Session retains
            # connection/log continuity and prevents use after a terminal fault.
            self._facts = _Facts(round_id=old.round_id + 1, self_hand=hand,
                                 remaining_counts=tuple(counts), level_rank=data["levelCard"])
            return
        _require(old.round_id > 0 and old.stage != "settled", "Event has no active beginning")
        _require(old.facts_status == "trusted", "Degraded facts cannot be repaired by an action request")
        if kind == "action_requested":
            new = self._requested(old, data)
        elif kind == "play_observed":
            new = self._played(old, data, event_id)
        elif kind == "tribute_announced":
            _require(old.last_play is None and not old.exchange_applied and not old.anti_tribute,
                     "Tribute arrived after play or exchange resolution")
            _require(not old.tribute_transfers, "Tribute mapping repeated without a new round")
            transfers = tuple(tuple(row) for row in data["result"])
            for sender, _, card in transfers:
                if sender == self.seat:
                    _remove(old.self_hand, (card,))
            new = replace(old, stage="tribute", tribute_transfers=transfers)
        elif kind == "exchange_completed":
            new = self._exchanged(old, data)
        elif kind == "anti_tribute":
            _require(old.last_play is None and not old.tribute_transfers and not old.exchange_applied
                     and not old.anti_tribute, "Anti-tribute conflicts with an existing exchange")
            new = replace(old, stage="waiting", anti_tribute=True)
        elif kind == "round_result":
            order = tuple(data["order"])
            known_order = tuple(seat for seat in order if seat in old.finished_order)
            _require(known_order == old.finished_order, "Settlement contradicts observed finishing order")
            _require(data["curRank"] == old.level_rank, "Settlement changed the current level rank")
            # restCards is deliberately never copied. It belongs to postgame
            # evidence and must not enrich this seat's past/future observations.
            new = replace(old, stage="settled", finished_order=order)
        else:
            raise StateConflict("Unsupported fact event")
        self._facts = new

    def _requested(self, old: _Facts, data: Mapping[str, Any]) -> _Facts:
        stage = data["stage"]
        _require(self.seat not in old.finished_order, "Finished player received an action request")
        hand = tuple(data["handCards"])
        _require(Counter(hand) == Counter(old.self_hand), "Request hand contradicts observed own transfers")
        _require(data["curRank"] == old.level_rank, "Request changed the round's current level rank")
        if stage in ("tribute", "back"):
            _require(old.last_play is None and not old.exchange_applied and not old.anti_tribute,
                     "Exchange request arrived after play or exchange resolution")
        if stage == "tribute":
            _require(not old.tribute_transfers, "Tribute request arrived after its announcement")
        elif stage == "back":
            expected = (data["tributePos"], self.seat, data["tribute"])
            _require(expected in old.tribute_transfers, "Back request lacks its announced tribute pairing")
        elif stage == "play":
            _require(not old.tribute_transfers or old.exchange_applied,
                     "Play request precedes completion of announced exchange")
            if data["curAction"] is not None:
                _require(old.last_play is not None
                         and old.last_play.seat == data["curPos"]
                         and _same_action(old.last_play.action, _action(data["curAction"])),
                         "Request context lacks its preceding public play")
        counts = old.remaining_counts
        if "publicInfo" in data:
            incoming = tuple(row["rest"] for row in data["publicInfo"])
            _require(all(known is None or known == actual for known, actual in zip(counts, incoming)),
                     "Request counts contradict observed public actions")
            _require(incoming[self.seat] == len(hand), "Request own count contradicts own hand")
            counts = incoming
        ranks: list[str | None] = [None, None]
        ranks[self.seat % 2], ranks[1 - self.seat % 2] = data["selfRank"], data["oppoRank"]
        leader, leading = old.leader_seat, old.leading_action
        if stage == "play":
            leader = data["greaterPos"] if data["greaterPos"] >= 0 else None
            leading = _action(data["greaterAction"]) if leader is not None else None
        return replace(old, stage=stage, self_hand=hand, remaining_counts=counts,
                       team_ranks=tuple(ranks), leader_seat=leader, leading_action=leading)

    def _played(self, old: _Facts, data: Mapping[str, Any], event_id: str) -> _Facts:
        _require(not old.tribute_transfers or old.exchange_applied,
                 "Play notification precedes completion of announced exchange")
        actor, action = data["curPos"], _action(data["curAction"])
        _require(actor not in old.finished_order and old.remaining_counts[actor] != 0,
                 "Finished player produced another action")
        counts, hand, finished = list(old.remaining_counts), old.self_hand, old.finished_order
        leader, leading = old.leader_seat, old.leading_action
        if action != _PASS:
            cards = tuple(action[2])
            if actor == self.seat:
                hand = _remove(hand, cards)
            if counts[actor] is not None:
                _require(counts[actor] >= len(cards), "Public action consumes more than the known count")
                counts[actor] -= len(cards)
                if counts[actor] == 0:
                    finished += (actor,)
            leader, leading = actor, action
        # Notify's greater* describes the state BEFORE this action. Keep the
        # last actual public lead here; the next act supplies its current lead.
        play = PublicPlay(event_id=event_id, seat=actor, action=action)
        history = old.history + (play,)
        trim = max(0, len(history) - self.history_limit)
        history = history[trim:]
        return replace(old, stage="play", self_hand=hand, remaining_counts=tuple(counts),
                       leader_seat=leader, leading_action=leading, finished_order=finished,
                       history=history, last_play=play, history_omitted=old.history_omitted + trim,
                       history_complete=old.history_complete and trim == 0)

    def _exchanged(self, old: _Facts, data: Mapping[str, Any]) -> _Facts:
        _require(old.tribute_transfers and not old.exchange_applied and not old.anti_tribute
                 and old.last_play is None, "Exchange lacks a pending tribute announcement")
        backs = tuple(tuple(row) for row in data["result"])
        _require({(sender, recipient) for sender, recipient, _ in backs}
                 == {(recipient, sender) for sender, recipient, _ in old.tribute_transfers},
                 "Back transfer mapping does not reverse the announced tributes")
        transfers = old.tribute_transfers + backs
        outgoing = tuple(card for sender, _, card in transfers if sender == self.seat)
        incoming = tuple(card for _, recipient, card in transfers if recipient == self.seat)
        hand = _remove(old.self_hand, outgoing) + incoming
        _require(len(hand) == data["count"], "Exchange count contradicts computed own hand")
        counts = list(old.remaining_counts)
        counts[self.seat] = len(hand)
        return replace(old, stage="waiting", self_hand=hand, remaining_counts=tuple(counts),
                       back_transfers=backs, exchange_applied=True)

    def visible(self, binding: RequestBinding) -> VisibleState:
        old = self._facts
        _require(binding.seat == self.seat and binding.round_id == old.round_id,
                 "Binding belongs to a different seat or round")
        _require(old.stage in ("play", "tribute", "back"), "No action stage for a decision snapshot")
        _require(old.facts_status == "trusted", "Cannot decide from degraded facts")
        return VisibleState(binding=binding, stage=old.stage, self_hand=old.self_hand,
                            level_rank=old.level_rank, remaining_counts=old.remaining_counts,
                            team_ranks=old.team_ranks, turn_seat=self.seat,
                            leader_seat=old.leader_seat, leading_action=old.leading_action,
                            finished_order=old.finished_order, history=old.history,
                            history_complete=old.history_complete, history_omitted=old.history_omitted,
                            facts_status=old.facts_status)

    def table_view(self) -> TableView | None:
        if not self.round_id:
            return None
        old = self._facts
        return TableView(stage=old.stage, self_hand=old.self_hand, level_rank=old.level_rank,
                         team_ranks=old.team_ranks, remaining_counts=old.remaining_counts,
                         leader_seat=old.leader_seat, leading_action=old.leading_action,
                         finished_order=old.finished_order, facts_status=old.facts_status)
