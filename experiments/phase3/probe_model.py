#!/usr/bin/env python3
"""Bounded LangGraph/DeepSeek experiment; never connects to or submits to a game.

Synthetic fixtures assert their candidate sets, not game-engine correctness.
No dotenv/config secrets are read. Live requests require an explicit --mode live.
Provider syntax references checked 2026-10-19:
https://api-docs.deepseek.com/guides/json_mode/
https://api-docs.deepseek.com/quick_start/pricing/
https://api-docs.deepseek.com/guides/thinking_mode/
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypedDict

# This standalone experiment never enables external tracing, even if inherited.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "model_cases.json"
PUBLIC_FIELDS = {
    "phase", "self_seat", "partner_seat", "level_rank", "my_hand",
    "remaining_counts", "leading_seat", "leading_action", "public_history",
    "history_complete",
}
CANDIDATE_FIELDS = {
    "action_id", "type", "cards", "remaining_count", "breaks_structure",
}


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=240)


@dataclass(frozen=True)
class ProbeConfig:
    total_budget_s: float = 30.0
    call_timeout_s: float = 12.0
    reserve_s: float = 0.2
    min_review_budget_s: float = 1.0
    max_calls: int = 2

    def __post_init__(self):
        if self.max_calls not in (1, 2):
            raise ValueError("max_calls must be 1 or 2")
        if min(self.total_budget_s, self.call_timeout_s, self.min_review_budget_s) <= 0:
            raise ValueError("time budgets must be positive")
        if not 0 <= self.reserve_s < self.total_budget_s:
            raise ValueError("reserve must be smaller than total budget")


@dataclass
class RequestLease:
    """Experiment-only request validity; not a server reconnection mechanism."""
    current_version: str
    active: bool = True
    expires_at: float | None = None

    def valid(self, snapshot_version: str, now: float) -> bool:
        return (self.active and self.current_version == snapshot_version
                and (self.expires_at is None or now < self.expires_at))


@dataclass
class Reply:
    content: str | None = None
    usage: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None


class Caller(Protocol):
    transport_requests: int

    async def __call__(self, role: str, payload: dict[str, Any], timeout_s: float) -> Reply: ...


class ProbeState(TypedDict, total=False):
    snapshot: dict[str, Any]
    started: float
    deadline: float
    calls: list[dict[str, Any]]
    events: list[dict[str, Any]]
    call_count: int
    review_used: bool
    main_decision: dict[str, str] | None
    suggestion: dict[str, str] | None
    last_error: str | None
    triggers: list[str]
    final_action_id: str | None
    source: str | None
    status: str


def public_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    """Explicit public-input projection; extra fixture fields never reach prompts."""
    obs = case.get("observation", {})
    candidates = case.get("candidates", [])
    return {
        "id": str(case["id"]),
        "request_version": str(case["request_version"]),
        "observation": copy.deepcopy({k: v for k, v in obs.items() if k in PUBLIC_FIELDS}),
        "candidates": copy.deepcopy([
            {k: v for k, v in candidate.items() if k in CANDIDATE_FIELDS}
            for candidate in candidates
        ]),
        "fallback_action_id": case.get("fallback_action_id"),
    }


def safe_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    aliases = {"input_tokens": ("input_tokens", "prompt_tokens"),
               "output_tokens": ("output_tokens", "completion_tokens"),
               "total_tokens": ("total_tokens",)}
    result = {}
    for target, names in aliases.items():
        for name in names:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                result[target] = value
                break
    return result or None


def parse_reply(reply: Reply, candidates: list[dict[str, Any]]) -> tuple[dict[str, str] | None, str | None]:
    try:
        if reply.parsed is not None:
            result = Decision.model_validate(reply.parsed)
        elif isinstance(reply.content, str):
            result = Decision.model_validate_json(reply.content)
        else:
            return None, "empty_response"
    except ValidationError:
        return None, "invalid_json_or_schema"
    if result.action_id not in {a["action_id"] for a in candidates}:
        return None, "invalid_candidate_id"
    return result.model_dump(), None


def risk_triggers(snapshot: dict[str, Any], decision: dict[str, str] | None) -> list[str]:
    if decision is None:
        return ["repair_invalid_output"]
    obs = snapshot["observation"]
    triggers = []
    for seat, count in enumerate(obs.get("remaining_counts", [])):
        if seat not in (obs.get("self_seat"), obs.get("partner_seat")) and type(count) is int and 1 <= count <= 2:
            triggers.append("opponent_near_finish")
            break
    action = next(a for a in snapshot["candidates"] if a["action_id"] == decision["action_id"])
    if action.get("type") == "bomb":
        triggers.append("use_bomb")
    if action.get("breaks_structure") is True:
        triggers.append("fixture_structure_cost")
    if obs.get("leading_seat") == obs.get("partner_seat") and action.get("type") != "pass":
        triggers.append("overtake_partner")
    return triggers


def _consume_task(task: asyncio.Task) -> None:
    """Discard cancelled/late results; never let them change the selected action."""
    try:
        task.exception()
    except BaseException:
        pass


class ModelProbe:
    def __init__(self, caller: Caller, config: ProbeConfig, lease: RequestLease):
        self.caller, self.config, self.lease = caller, config, lease
        graph = StateGraph(ProbeState)
        graph.add_node("prepare", self.prepare)
        graph.add_node("main", self.main)
        graph.add_node("review", self.review)
        graph.add_node("validate", self.validate)
        graph.add_node("fallback", self.fallback)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges("prepare", self.route_prepare, {"main": "main", "validate": "validate", "end": END})
        graph.add_conditional_edges("main", self.route_main, {"review": "review", "validate": "validate", "fallback": "fallback"})
        graph.add_edge("review", "validate")
        graph.add_conditional_edges("validate", self.route_validate, {"fallback": "fallback", "end": END})
        graph.add_edge("fallback", END)
        self.graph = graph.compile()

    def valid(self, state: ProbeState) -> bool:
        now = time.monotonic()
        return now < state["deadline"] and self.lease.valid(state["snapshot"]["request_version"], now)

    def available(self, state: ProbeState) -> float:
        deadline = min(state["deadline"], self.lease.expires_at or float("inf"))
        return deadline - time.monotonic() - self.config.reserve_s

    def prepare(self, state: ProbeState) -> dict[str, Any]:
        candidates = state["snapshot"]["candidates"]
        ids = [a.get("action_id") for a in candidates]
        if not self.valid(state):
            return {"status": "expired", "last_error": "request_expired"}
        if not candidates:
            return {"status": "empty_candidates", "last_error": "empty_fixture_candidates"}
        if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)) or state["snapshot"]["fallback_action_id"] not in ids:
            return {"status": "invalid_fixture", "last_error": "invalid_fixture_mapping"}
        if len(candidates) == 1:
            return {"suggestion": {"action_id": ids[0], "reason": "唯一输入候选，直接校验。"}, "source": "unique_candidate"}
        return {}

    def route_prepare(self, state: ProbeState) -> str:
        if state["status"] != "pending":
            return "end"
        return "validate" if state.get("suggestion") else "main"

    async def call(self, state: ProbeState, role: str) -> dict[str, Any]:
        if not self.valid(state):
            return {"suggestion": None, "last_error": "request_expired"}
        timeout_s = min(self.config.call_timeout_s, self.available(state))
        if timeout_s <= 0 or state["call_count"] >= self.config.max_calls:
            return {"suggestion": None, "last_error": "budget_exhausted"}
        payload = {"fixture_notice": "synthetic candidate set; not engine-verified", "snapshot": state["snapshot"]}
        if role == "review":
            payload.update(main_decision=state.get("main_decision"), triggers=state.get("triggers", []), previous_error=state.get("last_error"))
        started = time.monotonic()
        transport_before = self.caller.transport_requests
        record: dict[str, Any] = {"attempt": state["call_count"] + 1, "role": role, "usage": None}
        decision, error = None, None
        task = asyncio.create_task(self.caller(role, copy.deepcopy(payload), timeout_s))
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
            if not done:
                task.cancel()
                task.add_done_callback(_consume_task)
                error = "timeout"
            else:
                reply = task.result()
                record["usage"] = safe_usage(reply.usage)
                if not self.valid(state):
                    error = "request_expired"
                else:
                    decision, error = parse_reply(reply, state["snapshot"]["candidates"])
        except Exception as exc:
            # Never serialize str(exc), response bodies, headers, or credentials.
            record["error_type"] = type(exc).__name__
            error = "provider_error"
        record.update(status=error or "valid", latency_ms=round((time.monotonic() - started) * 1000, 3),
                      transport_requests=self.caller.transport_requests - transport_before)
        if decision:
            record["decision"] = decision
        return {"suggestion": decision, "last_error": error, "call_count": state["call_count"] + 1,
                "calls": state["calls"] + [record], "source": role}

    async def main(self, state: ProbeState) -> dict[str, Any]:
        update = await self.call(state, "main")
        decision = update.get("suggestion")
        update["main_decision"] = decision
        update["triggers"] = risk_triggers(state["snapshot"], decision)
        if update["triggers"] and self.config.max_calls == 2 and (self.available(state) < self.config.min_review_budget_s):
            update["events"] = state["events"] + [{"event": "review_skipped", "reason": "insufficient_time"}]
        return update

    def route_main(self, state: ProbeState) -> str:
        if not self.valid(state):
            return "validate"
        if state.get("last_error") in ("timeout", "provider_error", "budget_exhausted"):
            return "fallback"
        if (state.get("triggers") and self.config.max_calls == 2 and state["call_count"] < 2
                and not state["review_used"] and self.available(state) >= self.config.min_review_budget_s):
            return "review"
        return "validate"

    async def review(self, state: ProbeState) -> dict[str, Any]:
        update = await self.call(state, "review")
        update["review_used"] = update.get("call_count", state["call_count"]) > state["call_count"]
        if not update["review_used"] and self.valid(state):
            # Time can run out between routing and entering this node. A skipped
            # review must not erase an otherwise valid main decision.
            update.update(suggestion=state.get("main_decision"), source="main",
                          events=state["events"] + [{"event": "review_skipped", "reason": "budget_or_validity_changed"}])
        return update

    def validate(self, state: ProbeState) -> dict[str, Any]:
        if not self.valid(state):
            return {"status": "expired", "suggestion": None, "final_action_id": None, "last_error": "request_expired"}
        suggestion = state.get("suggestion")
        if suggestion and suggestion["action_id"] in {a["action_id"] for a in state["snapshot"]["candidates"]}:
            return {"status": "selected", "final_action_id": suggestion["action_id"]}
        return {"status": "needs_fallback"}

    def route_validate(self, state: ProbeState) -> str:
        return "fallback" if state["status"] == "needs_fallback" else "end"

    def fallback(self, state: ProbeState) -> dict[str, Any]:
        if not self.valid(state):
            return {"status": "expired", "final_action_id": None, "last_error": "request_expired"}
        fallback = state["snapshot"]["fallback_action_id"]
        if fallback not in {a["action_id"] for a in state["snapshot"]["candidates"]}:
            return {"status": "invalid_fixture", "final_action_id": None}
        return {"status": "fallback", "source": "rule_fallback", "final_action_id": fallback}

    async def run(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        snapshot = public_snapshot(case)
        initial: ProbeState = {"snapshot": snapshot, "started": started, "deadline": started + self.config.total_budget_s,
                               "call_count": 0, "calls": [], "events": [], "review_used": False,
                               "status": "pending", "final_action_id": None, "suggestion": None, "source": None}
        with tracing_context(enabled=False):
            final = await self.graph.ainvoke(initial, config={"callbacks": [], "recursion_limit": 12})
        return {"case_id": snapshot["id"], "request_version": snapshot["request_version"],
                "status": final["status"], "source": final.get("source"), "action_id": final.get("final_action_id"),
                "main_decision": final.get("main_decision"), "final_suggestion": final.get("suggestion"),
                "error": final.get("last_error"), "review_used": final["review_used"], "triggers": final.get("triggers", []),
                "model_attempts": final["call_count"], "transport_requests": sum(c["transport_requests"] for c in final["calls"]),
                "calls": final["calls"], "events": final["events"], "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "snapshot": snapshot, "platform_submission": "not_performed"}


class ScriptedCaller:
    """Offline responses: no network, no claimed provider latency or token usage."""
    def __init__(self, steps: list[dict[str, Any]], lease: RequestLease | None = None):
        self.steps = steps
        self.lease = lease
        self.invocations = 0
        self.transport_requests = 0

    async def __call__(self, role: str, payload: dict[str, Any], timeout_s: float) -> Reply:
        step = self.steps[self.invocations]
        self.invocations += 1
        await asyncio.sleep(step.get("delay_s", 0))
        if step.get("expire") and self.lease:
            self.lease.current_version = "new-request"
        if step.get("raise"):
            raise RuntimeError("scripted failure (message must not be logged)")
        return Reply(content=step.get("content"), usage=step.get("usage"))


class LiveCaller:
    """Only this class can reach DeepSeek; hidden transport retries are disabled."""
    def __init__(self, key: str, model: str, timeout_s: float, max_tokens: int, thinking: bool,
                 http_transport=None):
        import httpx
        from langchain_deepseek import ChatDeepSeek

        self.transport_requests = 0

        async def on_request(request):
            self.transport_requests += 1

        self.http = httpx.AsyncClient(timeout=timeout_s, transport=http_transport,
                                     event_hooks={"request": [on_request]})
        self.model = ChatDeepSeek(model=model, api_key=key, api_base="https://api.deepseek.com",
                                 max_retries=0, timeout=timeout_s, max_tokens=max_tokens,
                                 extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
                                 http_async_client=self.http)
        self.structured = self.model.with_structured_output(Decision, method="json_mode", include_raw=True)

    async def __call__(self, role: str, payload: dict[str, Any], timeout_s: float) -> Reply:
        from langchain_core.messages import HumanMessage, SystemMessage

        instruction = ("你是掼蛋决策探针的" + ("战术复核者" if role == "review" else "主决策者")
                       + "。仅使用提供的本人手牌和公开局势，未知信息保持未知。只选择 candidates 中一个 action_id。"
                       "不得猜出他人真实手牌，不得发起工具或提交动作。输出简短理由，不输出推理过程。"
                       '输出且仅输出 json 对象，例如 {"action_id":"pass","reason":"队友领先，保留控制牌。"}。'
                       "字段恰好为 action_id(string) 和 reason(1到240字符string)，无其他字段。"
                       "若是复核，可保留原选择或改选同一候选；此前无效时直接给出有效选择。")
        with tracing_context(enabled=False):
            result = await self.structured.ainvoke(
                [SystemMessage(content=instruction), HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
                config={"callbacks": []})
        raw = result.get("raw")
        usage = getattr(raw, "usage_metadata", None)
        if not usage:
            usage = getattr(raw, "response_metadata", {}).get("token_usage")
        parsed = result.get("parsed")
        return Reply(content=raw.content if raw and isinstance(raw.content, str) else None,
                     parsed=parsed.model_dump() if isinstance(parsed, Decision) else None, usage=usage)

    async def close(self):
        await self.http.aclose()


def summarize(results: list[dict[str, Any]], mode: str, model: str, config: ProbeConfig) -> dict[str, Any]:
    calls = [call for result in results for call in result["calls"]]
    totals = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [c["usage"][key] for c in calls if c["usage"] and key in c["usage"]]
        totals[key] = sum(values) if values else (0 if not calls else None)
    return {"status": "completed", "mode": mode, "model": model if mode == "live" else None,
            "case_count": len(results), "selected": sum(x["status"] == "selected" for x in results),
            "fallback": sum(x["status"] == "fallback" for x in results), "expired": sum(x["status"] == "expired" for x in results),
            "model_attempts": len(calls), "transport_requests": sum(x["transport_requests"] for x in results),
            "valid_responses": sum(c["status"] == "valid" for c in calls),
            "review_count": sum(x["review_used"] for x in results), "observed_token_totals": totals,
            "usage_missing_calls": sum(c["usage"] is None for c in calls), "fee": None,
            "call_latency_ms": [c["latency_ms"] for c in calls], "config": vars(config),
            "budget_scope": "experimental local budget, not verified platform deadline",
            "evidence_scope": "synthetic candidate validation and bounded graph only; no platform submission, hand-engine validation, or strength claim"}


async def cli(args) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    config = ProbeConfig(total_budget_s=args.total_budget, call_timeout_s=args.call_timeout,
                         reserve_s=args.reserve, min_review_budget_s=args.review_min_budget, max_calls=args.max_calls)
    if args.mode == "live" and not os.environ.get("DEEPSEEK_API_KEY"):
        return {"status": "blocked", "reason": "DEEPSEEK_API_KEY is not configured in the process environment",
                "mode": "live", "model": args.model, "model_attempts": 0, "transport_requests": 0}, [], 2
    data = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if not cases or set(args.case) - {c["id"] for c in cases}:
            raise ValueError("unknown case id")
    live = None
    results = []
    try:
        if args.mode == "live":
            live = LiveCaller(os.environ["DEEPSEEK_API_KEY"], args.model, args.call_timeout, args.max_tokens, args.thinking)
        for case in cases:
            lease = RequestLease(str(case["request_version"]))
            valid = json.dumps({"action_id": case["fallback_action_id"], "reason": "按固定模拟选择。"}, ensure_ascii=False)
            caller = live or ScriptedCaller([{"content": valid}, {"content": valid}], lease)
            results.append(await ModelProbe(caller, config, lease).run(case))
        summary = summarize(results, args.mode, args.model, config)
        summary.update(thinking=args.thinking if args.mode == "live" else None, max_tokens=args.max_tokens,
                       max_retries=0, structured_output_method="json_mode", fixture_notice=data.get("disclaimer"))
        return summary, results, 0
    finally:
        if live:
            await live.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("simulated", "live"), default="simulated")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--case", action="append")
    parser.add_argument("--total-budget", type=float, default=30)
    parser.add_argument("--call-timeout", type=float, default=os.environ.get("GUANDAN_PROBE_TIMEOUT_SECONDS", "20"))
    parser.add_argument("--reserve", type=float, default=0.2)
    parser.add_argument("--review-min-budget", type=float, default=1)
    parser.add_argument("--max-calls", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for summary.json and decisions.jsonl")
    args = parser.parse_args()
    try:
        summary, results, exit_code = asyncio.run(cli(args))
    except Exception as exc:
        summary, results, exit_code = {"status": "failed", "error_type": type(exc).__name__,
                                      "note": "exception body omitted to protect credentials"}, [], 1
    summary["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
