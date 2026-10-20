"""Deterministic, bounded candidate features over the platform's legal set.

Turn estimates are heuristic residual group counts, not optimal hand searches.
Only IDs in the complete platform list may become candidates.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace, dataclass
from typing import Any

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.contracts import (
    ActionFeatures, Candidate, CandidateSnapshot, ErrorInfo, LegalAction,
    RequestContext, ToolLimits, ToolResult, WireAction,
)
from .cheap import _BOMB_TYPES, _check_input, _strength, cheap_choice
from .java_engine import EngineError, ResidualEngine
from .summary import rank_counts as _rank_counts, remaining_cards


FINAL_RULE_VERSION = "b05-rules-v1"
CANDIDATE_VERSION = "b05-candidates-v1"


def _error(code: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="hand", safe_message={
        "TOOL_TIMEOUT": "Candidate preparation exceeded its deadline.",
        "ENGINE_UNAVAILABLE": "Residual analysis is unavailable for this run.",
        "CANDIDATE_OVERFLOW": "Required candidates exceed the configured limit.",
        "TOOL_INVALID_RESULT": "Candidate preparation could not produce a valid result.",
        "INTERNAL_ERROR": "Candidate preparation failed.",
    }[code], related_event_id=None)


def _residual(hand: tuple[str, ...], action: LegalAction) -> tuple[str, ...]:
    return remaining_cards(hand, action.wire_action)


def _estimate_groups(hand: tuple[str, ...], actions: tuple[WireAction, ...]) -> int:
    """One greedy partition of one first-play list; no recursive Java calls."""
    remaining = Counter(hand)
    groups = 0
    for action in sorted(actions, key=lambda action: -len(action[2])):
        cards = Counter(action[2])
        if action[0] != "PASS" and not cards - remaining:
            remaining.subtract(cards)
            groups += 1
    return groups + sum(remaining.values())


@dataclass(frozen=True)
class _Row:
    action: LegalAction
    residual: tuple[str, ...]
    features: ActionFeatures


class HandTools:
    cheap_choice = staticmethod(cheap_choice)

    def __init__(self, engine: ResidualEngine | None = None, *, clock: Clock | None = None) -> None:
        self.engine = engine
        self.clock = clock or SystemClock()
        self._deep_disabled = False
        self._pending: set[asyncio.Task[Any]] = set()
        self._cleanup: asyncio.Task[None] | None = None

    def _check_deadline(self, deadline: float) -> None:
        if self.clock.monotonic() >= deadline:
            raise TimeoutError

    def _disable_engine(self) -> None:
        self._deep_disabled = True
        if self.engine is not None and self._cleanup is None:
            self._cleanup = asyncio.create_task(self.engine.aclose())
            self._cleanup.add_done_callback(self._consume)

    @staticmethod
    def _consume(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    async def aclose(self) -> None:
        self._disable_engine()
        for task in tuple(self._pending):
            task.cancel()
        if self._cleanup is not None:
            # Real Java cleanup has its own bounded terminate/kill waits. The
            # extra cap also makes malformed injected engines harmless at close.
            done, _ = await asyncio.wait({self._cleanup}, timeout=1.0)
            if not done:
                self._cleanup.cancel()

    async def prepare(self, context: RequestContext, limits: ToolLimits) -> ToolResult:
        try:
            self._check_deadline(limits.deadline_mono)
            task = asyncio.create_task(self._prepare(context, limits))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
            task.add_done_callback(self._consume)
            done, _ = await asyncio.wait(
                {task}, timeout=max(0, limits.deadline_mono - self.clock.monotonic()),
            )
            if not done:
                task.cancel()
                self._disable_engine()
                return ToolResult(snapshot=None, error=_error("TOOL_TIMEOUT"))
            self._check_deadline(limits.deadline_mono)
            return task.result()
        except TimeoutError:
            self._disable_engine()
            return ToolResult(snapshot=None, error=_error("TOOL_TIMEOUT"))
        except asyncio.CancelledError:
            if "task" in locals():
                task.cancel()
            self._disable_engine()
            raise
        except Exception:
            self._disable_engine()
            return ToolResult(snapshot=None, error=_error("INTERNAL_ERROR"))

    def _features(self, context: RequestContext, action: LegalAction) -> _Row:
        state = context.state
        if state.stage in ("tribute", "back"):
            direction = 1 if state.stage == "tribute" else -1
            features = ActionFeatures(
                played_count=None, remaining_count=None, estimated_turns=None,
                single_count=None, split_cost=None, is_bomb=False, finishes_hand=False,
                beats_leader=False,
                rule_score=float(direction * _strength(action.wire_action[2][0], state.level_rank)),
            )
            return _Row(action, (), features)
        passing = action.wire_action[0] == "PASS"
        residual = _residual(state.self_hand, action)
        before = _rank_counts(state.self_hand)
        after = _rank_counts(residual)
        split = float(sum(
            (1 + (2 if count >= 4 else 0))
            for rank, count in before.items() if count >= 2 and 0 < after[rank] < count
        ))
        played_count = len(state.self_hand) - len(residual)
        bomb = action.wire_action[0] in _BOMB_TYPES
        features = ActionFeatures(
            played_count=played_count, remaining_count=len(residual),
            estimated_turns=len(after), single_count=sum(count == 1 for count in after.values()),
            split_cost=split, is_bomb=bomb, finishes_hand=not residual,
            beats_leader=not passing and state.leader_seat is not None,
            rule_score=0.0,
        )
        row = _Row(action, residual, features)
        return replace(row, features=features.model_copy(update={"rule_score": self._score(context, row)}))

    @staticmethod
    def _pressure(context: RequestContext) -> bool:
        state = context.state
        leader = state.leader_seat
        return (leader is not None and leader % 2 != state.binding.seat % 2
                and state.remaining_counts[leader] in (1, 2))

    def _score(self, context: RequestContext, row: _Row) -> float:
        state, action, f = context.state, row.action, row.features
        if state.stage != "play":
            return f.rule_score
        passing = action.wire_action[0] == "PASS"
        partner_leading = state.leader_seat == (state.binding.seat + 2) % 4
        strength_cost = 0.0 if passing else sum(_strength(card, state.level_rank) for card in action.wire_action[2]) * .05
        wildcard_cost = 0.0 if passing else action.wire_action[2].count(f"H{state.level_rank}") * 5.0
        score = (
            10000.0 * f.finishes_hand + 10.0 * f.played_count
            - 4.0 * f.estimated_turns - 1.5 * f.single_count - 6.0 * f.split_cost
            - 25.0 * f.is_bomb - strength_cost - wildcard_cost
        )
        if passing:
            score += 1000.0 if partner_leading else -25.0
        if self._pressure(context) and f.beats_leader:
            score += 250.0
        if state.leader_seat is None and passing:
            score -= 20000.0
        return float(score)

    @staticmethod
    def _rank(rows: list[_Row]) -> list[_Row]:
        return sorted(rows, key=lambda row: (-row.features.rule_score, row.action.source_indices[0]))

    def _required(self, context: RequestContext, ranked: list[_Row]) -> dict[str, list[str]]:
        reasons: dict[str, list[str]] = {}
        for row in ranked:
            if row.action.wire_action[0] == "PASS":
                reasons.setdefault(row.action.action_id, []).append("legal_pass")
            if row.features.finishes_hand:
                reasons.setdefault(row.action.action_id, []).append("finishes_hand")
        if self._pressure(context):
            first = next((row for row in ranked if row.features.beats_leader), None)
            if first is not None:
                reasons.setdefault(first.action.action_id, []).append("opponent_close")
        return reasons

    async def _prepare(self, context: RequestContext, limits: ToolLimits) -> ToolResult:
        try:
            _check_input(context.state, context.legal)
            rows = []
            for index, action in enumerate(context.legal.actions):
                self._check_deadline(limits.deadline_mono)
                rows.append(self._features(context, action))
                if index % 64 == 63:
                    await asyncio.sleep(0)
            ranked = self._rank(rows)
            if len(self._required(context, ranked)) > limits.candidate_hard_limit:
                return ToolResult(snapshot=None, error=_error("CANDIDATE_OVERFLOW"))
            degraded = []
            if limits.deep_eval_limit and context.state.stage == "play":
                if self.engine is None or self._deep_disabled or getattr(self.engine, "disabled", False):
                    degraded.append("ENGINE_UNAVAILABLE")
                else:
                    updated = {row.action.action_id: row for row in rows}
                    for row in ranked[:limits.deep_eval_limit]:
                        self._check_deadline(limits.deadline_mono)
                        if not row.residual:
                            continue
                        actions = await self.engine.analyze(row.residual, context.state.level_rank, limits.deadline_mono)
                        self._check_deadline(limits.deadline_mono)
                        if not actions or any(action[0] == "PASS" or Counter(action[2]) - Counter(row.residual) for action in actions):
                            raise EngineError()
                        turns = _estimate_groups(row.residual, actions)
                        f = row.features.model_copy(update={"estimated_turns": turns})
                        row = replace(row, features=f)
                        f = f.model_copy(update={"rule_score": self._score(context, row)})
                        updated[row.action.action_id] = replace(row, features=f)
                    ranked = self._rank(list(updated.values()))
            reasons = self._required(context, ranked)
            rule_id = ranked[0].action.action_id
            reasons.setdefault(rule_id, []).append("rule_choice")
            if len(reasons) > limits.candidate_hard_limit:
                return ToolResult(snapshot=None, error=_error("CANDIDATE_OVERFLOW"))
            count = max(limits.candidate_target, len(reasons))
            for row in ranked:
                if len(reasons) >= count:
                    break
                reasons.setdefault(row.action.action_id, []).append("top_score")
            self._check_deadline(limits.deadline_mono)
            candidates = tuple(Candidate(
                action_id=row.action.action_id, features=row.features,
                keep_reasons=tuple(reasons[row.action.action_id]),
            ) for row in ranked if row.action.action_id in reasons)
            snapshot = CandidateSnapshot(
                binding=context.state.binding,
                candidate_snapshot_id=context.state.binding.request_id + ":c1",
                candidates=candidates, rule_action_id=rule_id,
                tool_status="degraded" if degraded else "ok", degradation_codes=tuple(degraded),
            )
            return ToolResult(snapshot=snapshot, error=_error(degraded[0]) if degraded else None)
        except (TimeoutError, EngineError) as exc:
            self._disable_engine()
            code = "TOOL_TIMEOUT" if isinstance(exc, TimeoutError) else exc.code
            return ToolResult(snapshot=None, error=_error(code))
        except asyncio.CancelledError:
            raise
        except Exception:
            self._disable_engine()
            return ToolResult(snapshot=None, error=_error("TOOL_INVALID_RESULT"))
