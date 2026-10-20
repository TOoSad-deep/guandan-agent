"""P10 residual features, required candidates and bounded failure behavior."""
import asyncio
from collections import Counter

import pytest

from guandan_agent.clock import ManualClock, SystemClock
from guandan_agent.contracts import RequestContext, ToolLimits
from guandan_agent.hand import (
    CANDIDATE_VERSION, CHEAP_RULE_VERSION, FINAL_RULE_VERSION, EngineError, HandTools,
    cheap_choice,
)
from tests.unit.test_rules import PASS, make_state_and_legal


def context(hand, actions, **kwargs):
    state, legal = make_state_and_legal(hand, actions, **kwargs)
    return RequestContext(
        state=state, legal=legal, received_at="2026-10-19T02:00:00Z",
        fallback_action_id=cheap_choice(state, legal), policy="V1",
        platform_context={"tribute_pos": 2, "tribute_card": "SA"} if state.stage == "back" else None,
    )


def limits(clock, *, deep=0, target=12, hard=24, seconds=2.0):
    return ToolLimits(deadline_mono=clock.monotonic() + seconds, deep_eval_limit=deep,
                      candidate_target=target, candidate_hard_limit=hard)


def prepare(ctx, *, deep=0, target=12, hard=24, engine=None):
    clock = ManualClock()
    return asyncio.run(HandTools(engine, clock=clock).prepare(
        ctx, limits(clock, deep=deep, target=target, hard=hard),
    ))


def test_versions_preserve_recorded_b04_cheap_baseline():
    assert CHEAP_RULE_VERSION == "b02-cheap-v1"
    assert FINAL_RULE_VERSION == "b05-rules-v1"
    assert CANDIDATE_VERSION == "b05-candidates-v1"


def test_original_ids_required_pass_all_finishing_and_pressure_survive_top_one():
    hand = ["S9", "D9"]
    ctx = context(hand, [PASS, ("Pair", "9", hand), ("Pair", "9", hand),
                         ("TwoCards", "9", hand)], counts=[2, 1, None, None])
    result = prepare(ctx, target=1, hard=3)
    assert result.error is None
    candidates = {candidate.action_id: candidate for candidate in result.snapshot.candidates}
    assert set(candidates) == {"L0000", "L0001", "L0003"}
    assert "legal_pass" in candidates["L0000"].keep_reasons
    assert all("finishes_hand" in candidates[key].keep_reasons for key in ("L0001", "L0003"))
    assert any("opponent_close" in candidate.keep_reasons for candidate in candidates.values())


def test_required_overflow_returns_no_partial_snapshot():
    hand = ["S9", "D9"]
    result = prepare(context(hand, [PASS, ("Pair", "9", hand), ("AltPair", "9", hand)]), target=1, hard=2)
    assert result.snapshot is None
    assert result.error.code == "CANDIDATE_OVERFLOW"


def test_first_play_does_not_invent_pass_and_top_count_is_bounded():
    hand = ["S3", "D4", "C5", "H6"]
    ctx = context(hand, [("Single", card[1], [card]) for card in hand], leader=None)
    result = prepare(ctx, target=2, hard=2)
    assert len(result.snapshot.candidates) == 2
    assert result.snapshot.rule_action_id in {candidate.action_id for candidate in result.snapshot.candidates}
    assert all(candidate.action_id in {action.action_id for action in ctx.legal.actions}
               for candidate in result.snapshot.candidates)
    assert all("legal_pass" not in candidate.keep_reasons for candidate in result.snapshot.candidates)


def test_multicopy_residual_and_split_cost_use_multisets():
    ctx = context(["S9", "S9", "D4"], [("Single", "9", ["S9"]), PASS])
    result = prepare(ctx)
    feature = next(candidate.features for candidate in result.snapshot.candidates if candidate.action_id == "L0000")
    assert (feature.played_count, feature.remaining_count, feature.single_count, feature.estimated_turns) == (1, 2, 2, 2)
    assert feature.split_cost == 1.0
    assert ctx.state.self_hand == ("S9", "S9", "D4")


@pytest.mark.parametrize("opponent_count,pressure", [(None, False), (0, False), (1, True), (2, True), (3, False)])
def test_only_known_leading_opponent_triggers_required_suppression(opponent_count, pressure):
    ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS], counts=[2, opponent_count, 1, None])
    result = prepare(ctx, target=1, hard=2)
    assert any("opponent_close" in candidate.keep_reasons for candidate in result.snapshot.candidates) == pressure


def test_near_finished_partner_does_not_trigger_opponent_pressure():
    ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS], leader=2, counts=[2, 1, 1, 1])
    result = prepare(ctx)
    assert result.snapshot.rule_action_id == "L0001"
    assert all("opponent_close" not in candidate.keep_reasons for candidate in result.snapshot.candidates)


@pytest.mark.parametrize("stage", ["tribute", "back"])
def test_tribute_back_features_are_stage_specific_and_skip_java(stage):
    class Forbidden:
        async def analyze(self, *_):
            raise AssertionError("tribute/back must not use play estimation")
    ctx = context(["SA", "D3"], [(stage, stage, ["SA"]), (stage, stage, ["D3"])], stage=stage)
    result = prepare(ctx, deep=2, engine=Forbidden())
    for candidate in result.snapshot.candidates:
        f = candidate.features
        assert (f.played_count, f.remaining_count, f.estimated_turns, f.single_count, f.split_cost) == (None,) * 5
        assert not any((f.is_bomb, f.finishes_hand, f.beats_leader))
    assert result.snapshot.rule_action_id == ("L0000" if stage == "tribute" else "L0001")


class ResidualFake:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.closed = False

    async def analyze(self, hand, level_rank, deadline):
        self.calls.append((hand, level_rank, deadline))
        if self.fail:
            raise EngineError()
        if Counter(hand) == Counter(["S3", "S4", "S5", "S6", "S7"]):
            return (("Straight", "3", hand),)
        return tuple(("Single", card[1], (card,)) for card in hand)

    async def aclose(self):
        self.closed = True


def test_residual_first_play_combination_not_in_current_follow_actions_is_estimated_only():
    engine = ResidualFake()
    hand = ["S3", "S4", "S5", "S6", "S7", "SA"]
    ctx = context(hand, [("Single", "A", ["SA"]), PASS])
    result = prepare(ctx, deep=1, engine=engine)
    assert len(engine.calls) == 1
    assert engine.calls[0][0] == tuple(hand[:-1])
    single = next(c for c in result.snapshot.candidates if c.action_id == "L0000")
    assert single.features.estimated_turns == 1
    assert {c.action_id for c in result.snapshot.candidates} == {"L0000", "L0001"}
    assert all(a.wire_action[0] != "Straight" for a in ctx.legal.actions)


def test_same_visible_inputs_produce_same_features_and_engine_only_gets_own_residual():
    ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS])
    hidden_a, hidden_b = ["SA"], ["H2", "SB"]
    assert hidden_a != hidden_b
    first, second = ResidualFake(), ResidualFake()
    a, b = prepare(ctx, deep=2, engine=first), prepare(ctx, deep=2, engine=second)
    assert a == b
    assert first.calls == second.calls
    assert all(not (Counter(call[0]) - Counter(ctx.state.self_hand)) for call in first.calls)


def test_engine_failure_falls_back_now_then_cheap_degraded_without_retry():
    async def scenario():
        clock, engine = ManualClock(), ResidualFake(fail=True)
        tools = HandTools(engine, clock=clock)
        ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS])
        first = await tools.prepare(ctx, limits(clock, deep=2))
        second = await tools.prepare(ctx, limits(clock, deep=2))
        await tools.aclose()
        assert first.snapshot is None and first.error.code == "ENGINE_UNAVAILABLE"
        assert second.snapshot.tool_status == "degraded"
        assert second.snapshot.degradation_codes == ("ENGINE_UNAVAILABLE",)
        assert len(engine.calls) == 1 and engine.closed
    asyncio.run(scenario())


def test_late_result_after_timeout_cannot_restore_snapshot_or_restart_engine():
    async def scenario():
        class Late(ResidualFake):
            async def analyze(self, hand, level, deadline):
                self.calls.append((hand, level, deadline))
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return (("Single", hand[0][1], (hand[0],)),)
        release = asyncio.Event()
        clock, engine = SystemClock(), Late()
        tools = HandTools(engine, clock=clock)
        ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS])
        first = await tools.prepare(ctx, limits(clock, deep=1, seconds=.02))
        assert first.snapshot is None and first.error.code == "TOOL_TIMEOUT"
        second = await tools.prepare(ctx, limits(clock, deep=1))
        assert second.snapshot.tool_status == "degraded"
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(engine.calls) == 1
        assert first.snapshot is None
        await tools.aclose()
    asyncio.run(scenario())


def test_deadline_includes_cheap_work_and_prevents_any_deep_call(monkeypatch):
    clock, engine = ManualClock(), ResidualFake()
    tools = HandTools(engine, clock=clock)
    original = tools._features
    def costly(*args):
        clock.advance(.6)
        return original(*args)
    monkeypatch.setattr(tools, "_features", costly)
    ctx = context(["S9", "D4"], [("Single", "9", ["S9"]), PASS])
    result = asyncio.run(tools.prepare(ctx, limits(clock, deep=2, seconds=.5)))
    assert result.snapshot is None and result.error.code == "TOOL_TIMEOUT"
    assert engine.calls == []
