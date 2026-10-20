#!/usr/bin/env python3
"""Frozen archived-input replay: 25 additional calls, no live game or submissions.

Default mode validates only. Live mode reads an existing process credential;
it never loads .env, retries, resumes, or replenishes the fixed schedule.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from guandan_agent.contracts import ModelDecision, ProviderInput, ProviderMessage, ProviderReply
from guandan_agent.model.provider import ProviderPort, safe_error, unknown_usage

LIMIT = 25


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def validate_plan(plan: dict) -> None:
    if (type(plan) is not dict or type(plan.get("global_call_limit")) is not int
            or plan["global_call_limit"] != LIMIT or plan.get("model") != "deepseek-flash"
            or not isinstance(plan.get("cases"), list) or len(plan["cases"]) != LIMIT):
        raise ValueError("Plan requires exactly 25 cases and the fixed DeepSeek model")
    previous = {}
    for case in plan["cases"]:
        identity = case.get("case_id", "")
        group = case.get("group")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity) or identity in previous:
            raise ValueError("Case IDs must be unique safe filenames")
        if group not in {"main", "review", "repair", "offline"}:
            raise ValueError("Unknown case group")
        ceiling = 1600 if group == "offline" else 384
        if type(case.get("max_output_tokens")) is not int or not 1 <= case["max_output_tokens"] <= ceiling:
            raise ValueError("Output limit exceeds the authorized case ceiling")
        timeout = case.get("timeout_s")
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 20:
            raise ValueError("Case requires a finite timeout at most 20 seconds")
        if not isinstance(case.get("messages"), list) or not case["messages"]:
            raise ValueError("Case requires frozen messages")
        for message in case["messages"]:
            ProviderMessage.model_validate(message)
        candidates = case.get("expected", {}).get("candidate_ids")
        if group != "offline" and (not isinstance(candidates, list) or not candidates
                                    or any(not isinstance(i, str) for i in candidates)
                                    or len(set(candidates)) != len(candidates)):
            raise ValueError("Action case requires unique expected candidate IDs")
        dependency = case.get("dependency")
        if dependency is not None:
            prior = previous.get(dependency.get("main_case_id"))
            if group != "review" or prior is None or prior["group"] != "main":
                raise ValueError("Review dependency must name an earlier main case")
            fallback = ModelDecision.model_validate(dependency["fallback_main"])
            if fallback.action_id not in candidates or set(prior["expected"]["candidate_ids"]) != set(candidates):
                raise ValueError("Review dependency must use the same candidate set")
            payload = json.loads(case["messages"][-1]["content"], object_pairs_hook=_unique)
            if case["messages"][-1]["role"] != "user" or not isinstance(payload, dict) or "main" not in payload:
                raise ValueError("Dependent review requires a last user JSON main field")
        previous[identity] = case


def check_output(case: dict, reply: ProviderReply) -> dict:
    result = {"json_valid": False, "schema_valid": None, "candidate_member": None,
              "decision": None, "semantic_judgment": "not_assessed"}
    if reply.error is not None or not reply.content:
        return result
    try:
        raw = json.loads(reply.content, object_pairs_hook=_unique)
        result["json_valid"] = True
        if case["group"] == "offline":
            # Envelope only; the independent audit owns item/evidence/semantic validation.
            result["offline_envelope_valid"] = (isinstance(raw, dict) and set(raw) == {"items"}
                                                and isinstance(raw["items"], list) and 1 <= len(raw["items"]) <= 3)
        else:
            result["schema_valid"] = False
            decision = ModelDecision.model_validate(raw)
            result["schema_valid"] = True
            result["candidate_member"] = decision.action_id in case["expected"]["candidate_ids"]
            if result["candidate_member"]:
                result["decision"] = decision.model_dump(mode="json")
    except (ValueError, TypeError):
        pass
    return result


def _messages(case: dict, results: dict) -> tuple[list[dict], bool | None]:
    messages = json.loads(_json(case["messages"]))
    dependency = case.get("dependency")
    fallback_used = None
    if dependency:
        decision = results.get(dependency["main_case_id"], {}).get("technical", {}).get("decision")
        fallback_used = decision is None
        payload = json.loads(messages[-1]["content"])
        payload["main"] = decision or dependency["fallback_main"]
        # Preserve the production prompt's key order; only replace main.
        messages[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return messages, fallback_used


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def run_plan(plan: dict, output: Path, *, mode: str, provider: ProviderPort) -> dict:
    validate_plan(plan)
    if mode not in {"live", "scripted"}:
        raise ValueError("An explicit provider mode is required")
    plan = json.loads(_json(plan))
    output.mkdir(parents=True, exist_ok=False)  # Refuse even an empty preexisting directory.
    with (output / "exclusive.lock").open("x", encoding="utf-8") as lock:
        lock.write("Never resume this directory or replace its lock.\n")
    _write(output / "plan.json", plan)
    results: dict[str, dict] = {}
    started = 0
    stop_reason = "interrupted"

    def checkpoint() -> dict:
        replies = [row["reply"] for row in results.values()]
        usage = {}
        known = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [row["usage"][field] for row in replies]
            known[field] = sum(value for value in values if value is not None)
            usage[field] = sum(values) if values and all(value is not None for value in values) and len(values) == started else None
        counts = [row["transport_requests"] for row in replies]
        summary = {"mode": mode, "evaluation_kind": "archived_input_replay", "real_platform": False,
                   "global_call_limit": LIMIT, "calls_started": started, "results_recorded": len(results),
                   "stop_reason": stop_reason, "scheduled_cases": LIMIT, "remaining_cases": LIMIT - started,
                   "real_provider": mode == "live", "usage": usage, "known_usage_subtotal": known,
                   "transport_requests": sum(counts) if counts and all(v is not None for v in counts) and len(counts) == started else None,
                   "known_transport_requests": sum(v for v in counts if v is not None), "cost_amount": None,
                   "provider_errors": sum(row["error"] is not None for row in replies),
                   "action_schema_valid": sum(row["technical"]["schema_valid"] is True for row in results.values()),
                   "candidate_members": sum(row["technical"]["candidate_member"] is True for row in results.values()),
                   "semantic_judgment": "not_assessed", "plan_sha256": hashlib.sha256(_json(plan).encode()).hexdigest()}
        _write(output / "summary.json", summary)
        return summary

    with (output / "calls.jsonl").open("x", encoding="utf-8") as journal:
        def append(row: dict) -> None:
            journal.write(_json(row) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        try:
            checkpoint()
            for case in plan["cases"]:
                if started >= LIMIT:
                    raise RuntimeError("Fixed call budget exhausted")
                messages, fallback_used = _messages(case, results)
                request = {"call_id": case["case_id"], "model": plan["model"], "messages": messages,
                           "max_output_tokens": case["max_output_tokens"], "timeout_s": case["timeout_s"]}
                request_hash = hashlib.sha256(_json(request).encode()).hexdigest()
                _write(output / f"{case['case_id']}.input.json", request)
                started += 1  # Permanent entrance; failures and cancellation consume the slot.
                append({"event": "begin", "ordinal": started, "case_id": case["case_id"],
                        "request_sha256": request_hash, "dependency_fallback_used": fallback_used})
                checkpoint()
                start = time.monotonic()
                try:
                    request_input = ProviderInput(call_id=case["case_id"],
                        messages=tuple(ProviderMessage.model_validate(m) for m in messages),
                        deadline_mono=start + case["timeout_s"], max_output_tokens=case["max_output_tokens"])
                    async with asyncio.timeout(case["timeout_s"]):
                        reply = await provider.complete(request_input)
                except asyncio.CancelledError:
                    stop_reason = "cancelled"
                    raise
                except Exception:
                    reply = ProviderReply(call_id=case["case_id"], content=None, usage=unknown_usage(),
                        transport_requests=None, elapsed_ms=(time.monotonic() - start) * 1000,
                        error=safe_error("VALIDATION_CALL_FAILED", "Validation call failed; details are not exposed and no retry was attempted."))
                row = {"event": "result", "case_id": case["case_id"], "group": case["group"],
                       "ordinal": started, "request_sha256": request_hash, "dependency_fallback_used": fallback_used,
                       "reply": reply.model_dump(mode="json"), "technical": check_output(case, reply)}
                results[case["case_id"]] = row
                _write(output / f"{case['case_id']}.result.json", row)
                append(row)
                checkpoint()
                print(_json({"case": case["case_id"], "ordinal": started,
                             "error": reply.error.code if reply.error else None,
                             "transport_requests": reply.transport_requests,
                             "total_tokens": reply.usage.total_tokens}), flush=True)
                if reply.transport_requests is not None and reply.transport_requests > 1:
                    stop_reason = "transport_budget_violation"
                    break
                if reply.error and reply.error.safe_message in {
                    "Provider request failed with HTTP status 401.", "Provider request failed with HTTP status 402.",
                }:
                    stop_reason = "provider_auth_or_credit_failure"
                    break
            else:
                stop_reason = "schedule_completed"
        finally:
            try:
                async with asyncio.timeout(2):
                    await provider.aclose()
            finally:
                summary = checkpoint()
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/validation25/plan.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"), object_pairs_hook=_unique)
        validate_plan(plan)
        if not args.live:
            print(_json({"mode": "check", "cases": LIMIT, "starts_calls": False}))
            return 0
        if args.output is None or args.output.exists():
            raise ValueError("Live mode requires a new output directory")
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("Missing process model credential")
        from guandan_agent.model.deepseek import DeepSeekProvider
        provider = DeepSeekProvider(api_key=key, model=plan["model"])
        summary = asyncio.run(run_plan(plan, args.output, mode="live", provider=provider))
        return 0 if summary["stop_reason"] == "schedule_completed" else 1
    except KeyboardInterrupt:
        print('{"error":"VALIDATION_INTERRUPTED"}')
        return 130
    except Exception:
        print('{"error":"VALIDATION_SETUP_OR_STORAGE_FAILED","details":"No automatic retry or resume."}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
