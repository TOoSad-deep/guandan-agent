"""Deterministic report text from validated, seat-scoped recording projections."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from guandan_agent.contracts import ExecutionRecord, RequestContext
from guandan_agent.semantics import describe_action, describe_card, describe_public, public_summary


_STAGES = {"play": "出牌", "tribute": "进贡", "back": "还贡"}


def program_facts(context: RequestContext, execution: ExecutionRecord,
                  later_public: Sequence[Mapping[str, Any]], *,
                  summarize_later: bool = False) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Render exact known fields; no inferred hidden cards or exchange counts.

    Public inputs are the strict projections made by offline.review. The output
    deliberately does not reuse historical model/heuristic reasons.
    """
    state = context.state
    seat, partner = state.binding.seat, (state.binding.seat + 2) % 4
    opponents = "、".join(str(other) for other in range(4) if other not in {seat, partner})
    stage = _STAGES[state.stage]
    counts = "；".join(f"座位{index}={count if count is not None else '未知'}"
                      for index, count in enumerate(state.remaining_counts))
    levels = "；".join(f"队伍{index}（座位{index}、{index + 2}）={rank if rank is not None else '未知'}"
                      for index, rank in enumerate(state.team_ranks))
    hand = "、".join(describe_card(card, state.level_rank) for card in state.self_hand) or "空"
    at_time = [
        f"阶段：{stage}（{state.stage}）；本人座位{seat}，队友座位{partner}，对手座位{opponents}。",
        f"当时本人手牌共{len(state.self_hand)}张：{hand}。",
        f"当前级牌：{state.level_rank}；队伍等级：{levels}。等级不是名次。",
        f"当时剩余张数：{counts}。未知张数保持未知。",
    ]
    if state.leading_action is None:
        at_time.append("当时无待压的领先牌。此字段不表示此前三家均过牌。")
    else:
        at_time.append(f"当时领先者为座位{state.leader_seat}；待压牌：{describe_action(state.leading_action, state.level_rank)}。")
    if state.finished_order:
        at_time.append("当时已有的出完顺序：" + "、".join(f"座位{value}" for value in state.finished_order) + "。")
    if context.platform_context is not None:
        exchange = context.platform_context
        at_time.append(f"平台还贡上下文：进贡方座位{exchange.tribute_pos}，进贡牌为{describe_card(exchange.tribute_card, state.level_rank)}。")
    if state.facts_status != "trusted":
        at_time.append("该请求的可见事实标记为降级（degraded），不能补全缺失信息。")
    at_time.append(f"原始合法动作去重后共{len(context.legal.actions)}项；未读取其他玩家暗牌。")
    # A submitted/confirmed choice is known from execution records, so keep it
    # after the request-time facts even when no public settlement is available.
    later = [f"本请求记录的{stage}选择：{execution.action_id}，{describe_action(execution.wire_action, state.level_rank)}；执行状态={execution.status}。"]
    if state.stage in {"tribute", "back"}:
        later.append(f"该动作表示{stage}选牌。本复盘公开投影未包含换牌完成后的本人手牌，不能按普通出牌减一计算换牌后张数。")
    for row in later_public:
        describe = public_summary if summarize_later else describe_public
        later.extend(describe(row["kind"], row["facts"], seat, state.level_rank))
    kinds = {row["kind"] for row in later_public}
    if "round_result" not in kinds:
        later.append("保留证据中没有本小局结算，不能确定本小局名次。")
    if "match_result" not in kinds:
        later.append("保留证据中没有本场结算，不能确定本场胜负。")
    return tuple(at_time), tuple(later)
